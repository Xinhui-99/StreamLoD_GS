#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.


import torch
import math
import diff_gaussian_rasterization 
#import diff_gaussian_rasterization_confidence
#import diff_gaussian_rasterization_c
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.cameras import SequentialCamera
from einops import repeat


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel,  visible_mask=None, iteration=0, iterations=100, is_training=False,  ape_code=-1):
    ## view frustum filtering for acceleration     
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc.get_anchor_feat[visible_mask]
    level = pc.get_level[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    grid_offsets = pc.get_offset[visible_mask]
 
    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    # get offset's opacity
    neural_opacity, scale_rot, color = pc.get_unified_mlp(cat_local_view_wodist)
    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1]) # [N_visible*n_offsets，1]
    mask = (neural_opacity>0.0) # [N_visible*n_offsets]
    mask = mask.view(-1)
    # select opacity 
    opacity = neural_opacity[mask]
    # get offset's color
    color = color.reshape([anchor.shape[0]*pc.n_offsets,((pc.max_sh_degree + 1) ** 2)*3])# [N_visible*n_offsets, 3*（8+1）]
    # get offset's cov
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [N_visible*n_offsets, 7]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [N_visible*n_offsets, 3]
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1) #[N_valid, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)# [N_visible*n_offsets, 9]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1) # [N_visible*n_offsets, 9+3*（8+1）+7+3]
    masked = concatenated_all[mask]
    dim_color = ((pc.max_sh_degree + 1) ** 2)*3 #3 * ((pc.max_sh_degree + 1) ** 2 )
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, dim_color, 7, 3], dim=-1)
    color_end =  color.reshape([-1,((pc.max_sh_degree + 1) ** 2), 3])
    
    # post-process cov # 处理缩放参数：基础缩放 × sigmoid激活的缩放调制
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    # 处理旋转参数
    rot = pc.rotation_activation(scale_rot[:,3:7])
    # post-process offsets to get centers for gaussians
    # 计算实际偏移：偏移向量 × 锚点的缩放参数
    offsets = offsets * scaling_repeat[:,:3]
    # 生成最终的高斯点位置：锚点位置 + 缩放后的偏移
    xyz = repeat_anchor + offsets 

    if is_training:
        neural_scaling_repeat, _, _, neural_scale_rot, _ = concatenated_all.split([6, 3, dim_color, 7, 3], dim=-1)
        neural_scaling = neural_scaling_repeat[:,3:] * torch.sigmoid(neural_scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
        level = level.repeat_interleave(pc.n_offsets, dim=0)   # [N*k]
        level_sel = level[mask]                    # [M]
        level_sel = level_sel.squeeze(-1)  # [M]，移除多余的维度
        level_sel = level_sel.to(torch.int64)  # 确保是 long 类型
        
        # 1) 拿到 level_sel 与 opacity（[M, C]）之后，生成每行的概率 p（[M]）
        #    注意：把 base_p 放到模块外或 register_buffer，避免每步重建
        #base_p = torch.tensor([0.1, 0.1, 0.10, 0.1], device=opacity.device, dtype=torch.float32) 0.1 0.15 0.2 0.25
        base_p = torch.tensor([0.1, 0.15, 0.25, 0.3], device=opacity.device, dtype=torch.float32)
        #base_p = torch.tensor([0.025, 0.05, 0.10, 0.20], device=opacity.device, dtype=torch.float32)

        scale = float(iteration) / float(iterations) 
        p = base_p.index_select(0, level_sel.long()).mul_(scale)   # [M]
        p.clamp_(0.0, 0.99)                                        # 原地裁剪，防 1-p 为 0

        # 2) 逐行伯努利采样（与 p 同形状），避免用 opacity.shape[0] 造成形状不一致
        rand = torch.empty_like(p).uniform_()                      # [M], U(0,1)
        keep = (rand > p).to(opacity.dtype)                        # [M], {0,1}

        # 3) 期望不变补偿；全部原地，减少中间张量
        denom = (1.0 - p).clamp_min_(1e-8)                         # [M]
        keep.div_(denom)                                           # keep -> compensation

        # 4) 应用到 opacity（原地广播乘）
        opacity.mul_(keep.unsqueeze_(1))                           # [M, C]
                         

        return xyz, color_end, opacity, scaling, rot, neural_opacity, neural_scaling, mask 
    else:
        return xyz, color_end, opacity, scaling, rot,  mask   

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
        image_shape = None, grad=False, is_train=False, iteration=0, iterations=100, frame_idx=1, visible_mask_save=None, is_report=None, visible_mask=None, retain_grad=False, ape_code=-1):

    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
    #is_training = pc.get_color_mlp.training
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    
    #mask_save = None
    if visible_mask_save is not None:
        with torch.inference_mode():
            pc.generate_neural_gaussians_prev(viewpoint_camera, visible_mask=visible_mask_save, is_train=is_train, grad=False)
        if is_report is not None or grad:
            xyz = pc.xyz_save 
            color_all = pc.color_all_save 
            opacity = pc.opacity_save 
            scaling = pc.scaling_save 
            rot = pc.rot_save 
        else:
            if is_train:
                xyz_n, color_all_n, opacity_n, scaling_n, rot_n, neural_opacity_n, neural_scaling_n, mask_n = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, iteration, iterations, is_training=is_train)
            else:
                xyz_n, color_all_n, opacity_n, scaling_n, rot_n, mask_n = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, iteration, iterations, is_training=is_train, ape_code=-1)

            # 直接使用torch.cat合并
            xyz = torch.cat([xyz_n, pc.xyz_save], dim=0)
            color_all = torch.cat([color_all_n, pc.color_all_save], dim=0)
            opacity = torch.cat([opacity_n, pc.opacity_save], dim=0)
            scaling = torch.cat([scaling_n, pc.scaling_save], dim=0)
            rot = torch.cat([rot_n, pc.rot_save], dim=0)
            mask = torch.cat([mask_n, pc.mask_save], dim=0)
            if is_train:
                neural_opacity = torch.cat((neural_opacity_n, pc.neural_opacity_save), dim=0)
                neural_scaling = torch.cat((neural_scaling_n, pc.neural_scaling_save), dim=0)
    else:
        if is_train:
            xyz_n, color_all_n, opacity_n, scaling_n, rot_n, neural_opacity_n, neural_scaling_n, mask_n = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, iteration, iterations, is_training=is_train)
            xyz = xyz_n
            color_all = color_all_n
            opacity = opacity_n
            scaling = scaling_n
            rot = rot_n
            neural_opacity = neural_opacity_n
            neural_scaling = neural_scaling_n
            mask = mask_n
        else:
            xyz_n, color_all_n, opacity_n, scaling_n, rot_n, mask_n = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, iteration, iterations, is_training=is_train, ape_code=-1)
            xyz = xyz_n
            color_all = color_all_n
            opacity = opacity_n
            scaling = scaling_n
            rot = rot_n
            mask = mask_n

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass
 
    color = color_all[:,0,:].squeeze() #:((pc.active_sh_degree + 1) ** 2)

    if image_shape is None:
        image_shape = (3, viewpoint_camera.image_height, viewpoint_camera.image_width)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
        image_height=image_shape[1],
        image_width=image_shape[2],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree= 1, #pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    #confidence=confidence,
    rasterizer = diff_gaussian_rasterization.GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    #opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = scaling  
        rotations = rot  

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer
    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    #generate_neural_gaussians_prev(self, viewpoint_camera, visible_mask=None)
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,#shs,color,#shs,
        colors_precomp = color,  #color, #colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)  
    # 基于alpha生成掩码
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    if is_train:
        if frame_idx==1: 
            return {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "selection_mask": mask,
                    "neural_opacity": neural_opacity,
                    "scaling": neural_scaling,
                    "visibility_filter" : radii > 0,
                    }
        else:
            return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "selection_mask": mask_n,
                "neural_opacity": neural_opacity_n,
                "scaling": neural_scaling_n,
                "visibility_filter" : radii > 0,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                }
 

def render_grad(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
        image_shape = None, is_report=None, retain_grad=False, ape_code=-1):

    """
    Render the scene. 
    Background tensor (bg_color) must be on GPU!
    """
    #is_training = pc.get_color_mlp.training
        
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    #mask_save = None
    pc.generate_neural_gaussians_prev(viewpoint_camera, grad=True)

    screenspace_points = torch.zeros_like(pc.xyz_save, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0


    color = pc.color_all_save[:,0,:].squeeze() #:((pc.active_sh_degree + 1) ** 2)
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if image_shape is None:
        image_shape = (3, viewpoint_camera.image_height, viewpoint_camera.image_width)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = diff_gaussian_rasterization.GaussianRasterizationSettings(
        image_height=image_shape[1],
        image_width=image_shape[2],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree= 1,#pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug 
    )

    #confidence=confidence,
    rasterizer = diff_gaussian_rasterization.GaussianRasterizer(raster_settings=raster_settings)
 
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.

    #cov3D_precomp = None
 
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer
    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    #generate_neural_gaussians_prev(self, viewpoint_camera, visible_mask=None)
    means3D = pc.xyz_save.detach().clone().requires_grad_(True)
    opacity = pc.opacity_save.detach().clone().requires_grad_(True)
    scaling = pc.scaling_save.detach().clone().requires_grad_(True)
    rot = pc.rot_save

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = screenspace_points,
        shs = None,#shs,color,#shs,
        colors_precomp = color,  #color, #colors_precomp,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)  
    # 基于alpha生成掩码
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
 
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "selection_mask_save": pc.mask_save,
            "xyz_save": pc.anchors_save,
            }


