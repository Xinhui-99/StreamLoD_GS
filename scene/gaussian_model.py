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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import math
import pickle
from datetime import timedelta
from functools import reduce
from torch_scatter import scatter_max
from sklearn.mixture import GaussianMixture
import copy
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler, PowerTransformer

from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal, getWorld2View2, getProjectionMatrix
from collections import OrderedDict
import torch.nn.functional as F
from utils.general_utils import strip_symmetric, build_scaling_rotation, warp_depth
from utils.image_utils import coords_grid
from utils.graphics_utils import knn_gpu
from utils.compress_utils import CompressedLatents, init_latents
from arguments import QuantizeParams, ModelParams
from scene.decoders import LatentDecoder, DecoderIdentity, DecoderLayer, LatentDecoderRes, Gate
import time
from scene.embedding import Embedding
from einops import repeat
  

class CompactUnifiedMLP(nn.Module):
    def __init__(self, feat_dim, view_dim, n_offsets, sh_degree):
        super().__init__()
        self.n_offsets = n_offsets
        
        # 计算输出维度
        self.splits = [
            n_offsets,                                    # opacity
            7 * n_offsets,                                # cov
            3 * ((sh_degree + 1) ** 2) * n_offsets       # color
        ]
        total_out = sum(self.splits)
        
        # 单个统一的网络
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + view_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, total_out)
        ).cuda()
        
    def forward(self, x):
        output = self.mlp(x)
        
        # 分割并应用激活
        outputs = torch.split(output, self.splits, dim=-1)
        
        return (
            torch.tanh(outputs[0]),      # opacity with tanh
            outputs[1],                   # cov without activation
            torch.sigmoid(outputs[2])     # color with sigmoid
        )
    
class MultiOptimizer:
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        
    def step(self):
        for opt in self.optimizers:
            opt.step()
    
    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)
            
    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]
    
    def load_state_dict(self, state_dicts):
        for opt, state_dict in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state_dict)
            
    @property
    def param_groups(self):
        param_groups = []
        for opt in self.optimizers:
            param_groups.extend(opt.param_groups)
        return param_groups

class GaussianModel:
    """3D Gaussian Splatting model with quantized latent representations and temporal consistency."""

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int,
                    latent_args: QuantizeParams, 
                    model_args: ModelParams, 
                    frame_idx : int = 1, 
                    use_xyz_legacy: bool = False,
                    feat_dim: int=32, 
                    n_offsets: int=5, 
                    fork: int=2,
                    grad_threhold: float = 0.0001,
                    gmm_threhold: float = 10,
                    use_feat_bank : bool = False,
                    appearance_dim : int = 32,
                    add_opacity_dist : bool = False,
                    add_cov_dist : bool = False,
                    add_color_dist : bool = False,
                    add_level: bool = False,
                    visible_threshold: float = -1,
                    dist2level: str = 'round',
                    grad_threthold_mode: str = 'Gmm',
                    base_layer: int = 10,
                    progressive: bool = True,
                    extend: float = 1.1
                     ):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #dtype = torch.float32 

        self.xyz_save = None
        self.color_all_save = None
        self.opacity_save = None
        self.scaling_save = None
        self.rot_save = None
        self.offset_save = None
        self.mask_save = None
        self.anchors_save = None

        self.feat_dim = feat_dim
        self.view_dim = 3
        self.n_offsets = n_offsets
        self.fork = fork
        self.grad_threhold = grad_threhold
        self.gmm_threhold = gmm_threhold
        self.grad_threthold_mode = grad_threthold_mode

        self.use_feat_bank = use_feat_bank
        self.grad_levels = 2
        self.diff_mask = None

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.add_level = add_level #false
        self.progressive = progressive # true

        # Octree
        self.sub_pos_offsets = torch.tensor([[i % fork, (i // fork) % fork, i // (fork * fork)] for i in range(fork**3)]).float().cuda()
        self.extend = extend
        self.visible_threshold = visible_threshold
        self.dist2level = dist2level
        self.base_layer = base_layer
        
        self.start_step = 0
        self.end_step = 0
        self._anchor_mask_level = None

        #self._anchor = torch.empty(0)
        self._level = torch.empty(0)
        #self._offset = torch.empty(0)
        #self._anchor_feat = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self._latents = OrderedDict([(n,torch.empty(0)) for n in latent_args.param_names])

        
        self.lod_rotation = torch.empty(0)
        #self._scaling = torch.empty(0)
        self.lod_opacity = torch.empty(0)
        self.scales_with_flow = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.param_names = latent_args.param_names
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.setup_decoders(latent_args)

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.level_dim = 1 if self.add_level else 0

        # 添加静态MLP存储（第一帧后会被初始化）
        self.unified_mlp_static = None

        # 替换原来的三个独立MLP
        # 使用统一的CompactUnifiedMLP
        self.unified_mlp = CompactUnifiedMLP(
            feat_dim=self.feat_dim,
            view_dim=self.view_dim, 
            n_offsets=self.n_offsets,
            sh_degree=sh_degree
        )
        self.unified_mlp_static = CompactUnifiedMLP(
            feat_dim=self.feat_dim,
            view_dim=self.view_dim, 
            n_offsets=self.n_offsets,
            sh_degree=sh_degree
        )
  
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.use_xyz_legacy = use_xyz_legacy
        print(f"Using xyz_legacy mode: {self.use_xyz_legacy}")
        # Initialize latent parameter storage
        self._latents = OrderedDict([(n,torch.empty(0)) for n in latent_args.param_names])

        # Gaussian tracking and optimization state
        self.max_radii2D_lod = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.anchor_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_lod = torch.empty(0)
        
        # Attribute masks for selective updates
        self.init_probs = None
        self.added_mask = None

        # Random number generator for splitting operations
        self.split_generator = torch.Generator(device="cuda")
        self.split_generator.manual_seed(latent_args.seed)

        self.param_names = latent_args.param_names
        self.mapping = None

        # Previous frame attributes
        self.prev_atts = OrderedDict({param_name:None for param_name in self.param_names})
        self.prev_latents = OrderedDict({param_name:None for param_name in self.param_names})
        self.prev_atts_initial = OrderedDict({param_name:None for param_name in self.param_names})
        self.frame_idx = frame_idx

        # Freeze states for different attributes
        self.frz_xyz = "none"
        self.frz_anchor = "none"
        self.frz_feature_anchor = "none"
        self.frz_features_dc =  "none"
        self.frz_features_rest =  "none"
        self.frz_scaling =  "none"
        self.frz_rotation =  "none"
        self.frz_opacity =  "none"
        self.gate_atts = None
        self.latent_args = latent_args
        self.model_args = model_args
        self.setup_functions()
        self.setup_decoders(latent_args)

    def setup_decoders(self, latent_args: QuantizeParams, verbose=False):
        """Initialize latent decoders for each Gaussian attribute based on quantization settings."""
        self.feature_dims = OrderedDict([
            ("anchor", 3),
            ("f_feature_anchor", 32),
            ("flow", 3),
            ("sc", 6),
            ("offset", 3*10)
        ])
        self.latent_decoders = OrderedDict()
        for i, param_name in enumerate(self.param_names):
            self.latent_decoders[param_name] = DecoderIdentity()
            if latent_args.quant_type[i] == 'sq':
                self.latent_decoders[param_name] = LatentDecoder(
                    latent_dim=latent_args.latent_dim[i],
                    feature_dim=self.feature_dims[param_name],
                    ldecode_matrix=latent_args.ldecode_matrix[i],
                    latent_norm=latent_args.latent_norm[i],
                    num_layers_dec=latent_args.num_layers_dec[i],
                    hidden_dim_dec=latent_args.hidden_dim_dec[i],
                    activation=latent_args.activation[i],
                    use_shift=latent_args.use_shift[i],
                    ldec_std=latent_args.ldec_std[i],
                    final_activation=latent_args.final_activation[i],
                ).cuda()
            if verbose:
                print(f"GaussianModel: Created {latent_args.quant_type[i]} decoder for {param_name}")

    def capture(self):
        return (
            self.active_sh_degree,
            self._latents,
            self.max_radii2D_lod,
            self.xyz_gradient_accum,
            self.anchor_gradient_accum,
            self.infl_accum,
            self.denom,
            self.denom_lod,
            self.infl_denom,
            OrderedDict([(n, l.state_dict()) for n,l in self.latent_decoders.items()]),
            self.gate_atts.state_dict() if self.gate_atts is not None else None,
            self.prev_atts,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.frame_idx,
            self.get_masks, 
            self._anchor,
            self._level,
            self._offset,
            self._local,
            self._lod_scaling,
            self.lod_rotation,
            self.lod_opacity,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._level,
        self._offset,
        self._local,
        self._lod_scaling, 
        self.lod_rotation, 
        self.lod_opacity,
        self._latents,
        self.max_radii2D_lod,
        xyz_gradient_accum, 
        anchor_gradient_accum,
        infl_accum, 
        denom,
        denom_lod,
        ldec_dicts,
        gate_att_dict,
        prev_atts,
        opt_dict, 
        self.spatial_lr_scale, 
        self.frame_idx,
        mask_atts) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.anchor_gradient_accum = anchor_gradient_accum
        self.infl_accum = infl_accum
        self.denom = denom
        self.denom_lod = denom_lod
        self.optimizer.load_state_dict(opt_dict)
        for n in ldec_dicts:
            self.latent_decoders[n].load_state_dict(ldec_dicts[n])
 
        if self.gate_atts is not None:
            self.gate_atts.load_state_dict(gate_att_dict)
        self.prev_atts = prev_atts

    def restore_fps(self, model_args, training_args, start_frame_idx):
        (self.active_sh_degree, 
        self._latents,
        self.max_radii2D_lod,
        xyz_gradient_accum, 
        anchor_gradient_accum,
        infl_accum, 
        denom,
        denom_lod,
        infl_denom,
        ldec_dicts,
        gate_att_dict,
        prev_atts,
        opt_dict, 
        self.spatial_lr_scale, 
        self.frame_idx,
        mask_atts) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.anchor_gradient_accum = anchor_gradient_accum
        self.infl_accum = infl_accum
        self.denom = denom
        self.denom_lod = denom_lod
        if start_frame_idx>1:
            self.frame_idx = 2
            self.update_residuals()
        for n in ldec_dicts:
            self.latent_decoders[n].load_state_dict(ldec_dicts[n])

        if self.gate_atts is not None:
            self.gate_atts.load_state_dict(gate_att_dict)
        self.prev_atts = prev_atts
    
    @property
    def get_atts(self):
        return self._latents
    
    @property
    def get_decoded_atts(self):
        return OrderedDict({"anchor"  : self._anchor,
                            "f_feature_anchor"  : self._anchor_feat,
                              "flow"  : self._flow ,
                              "sc"  : self._lod_scaling,
                              "offset"  : self._offset
                               } )

    @property
    def _ungated_anchor_res(self):
        """Get ungated xyz residual for regularization."""
        anchor = self.latent_decoders["anchor"](self._latents["anchor"])
        return anchor-self.prev_atts["anchor"]
 
    @property
    def _anchor_feat(self):
        """Get higher-order spherical harmonics features with optional gating."""
        if isinstance(self.latent_decoders["f_feature_anchor"], DecoderIdentity):
            anchor_feat = self._latents["f_feature_anchor"]
        else:
            anchor_feat = self.latent_decoders["f_feature_anchor"](self._latents["f_feature_anchor"])
            anchor_feat = anchor_feat.reshape(anchor_feat.shape[0], self.feat_dim) #self.feat_dim
        return anchor_feat  # shape (N, C-1, 3)

    @property
    def _flow(self):
        """Get optical flow vectors with optional gating."""
        flow = self.latent_decoders["flow"](self._latents["flow"])
        return flow

    @property
    def _offset(self):
        if isinstance(self.latent_decoders["offset"], DecoderIdentity):
            offset = self._latents["offset"]
            #offset = offset.reshape(offset.shape[0], self.n_offsets, 3)
        else:
            offset = self.latent_decoders["offset"](self._latents["offset"])
            offset = offset.reshape(offset.shape[0], 3, self.n_offsets)
        return offset
    
    @property
    def _anchor(self):
        # Decode latents to get xyz attribute
        anchor = self.latent_decoders["anchor"](self._latents["anchor"])
        # Apply gating if previous frame attributes exist and gating is enabled
        return anchor
    
    @property
    def _lod_scaling(self):
        """Get scaling parameters with optional gating."""
        lod_scaling = self.latent_decoders["sc"](self._latents["sc"])
        return lod_scaling
    
    @property
    def mask_cov(self):
        return torch.logical_or(self.mask_scaling, self.mask_rotation)

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def mask_color(self):
        return torch.logical_or(self.mask_features_dc, self.mask_features_rest)

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_flow(self):
        return self._flow

    @property
    def get_offset(self):
        return self._offset

    @property
    def get_extra_level(self):
        return self._extra_level

    @property
    def get_level(self):
        return self._level

    @property
    def get_scaling(self):
        # 前6维使用指数激活，后3维保持线性（flow）
        return self.scaling_activation(self._lod_scaling)
 
 
    @property
    def get_anchor_feat(self):
        return self._anchor_feat
 

    @property
    def get_unified_mlp(self):
        return self.unified_mlp   
 
 
    @property
    def get_opacity(self):
        return self.opacity_activation(self.lod_opacity)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
    
    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, dist_ratio)
                dist_min = torch.quantile(dist, 1 - dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - dist_ratio)
        self.standard_dist = dist_max
        if levels == -1:
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        else:
            self.levels = levels
        if init_level == -1:
            self.init_level = int(self.levels/2) #/2
        else:
            self.init_level = init_level

    def set_coarse_interval(self, coarse_iter, coarse_factor):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/coarse_factor
            a1 = coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_coarse_interval_grad(self, coarse_iter=1, coarse_factor=1.5):
        self.coarse_intervals_grad = []
        num_level_grad  = self.grad_levels - 1 - self.init_level_grad 
        if num_level_grad  > 0:
            q = 1/coarse_factor
            a1 = coarse_iter*(1-q)/(1-q**num_level_grad)
            temp_interval = 0
            for i in range(num_level_grad):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals_grad.append(interval)

    def parameters(self):
        return list(self._latents.values()) + \
                [param for decoder in self.latent_decoders.values() for param in list(decoder.parameters())]
    
    def named_parameters(self):
        parameter_dict = self._latents
        for n, decoder in self.latent_decoders.items():
            parameter_dict.update(
                {n+'.'+param_name:param for param_name, param in dict(decoder.named_parameters()).items()}
                )
        return parameter_dict
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @torch.inference_mode()
    def generate_neural_gaussians_save(self):
                    
        # 1. 使用 .data 访问底层tensor（最快）
        self.get_anchor_before = self.get_anchor.data
        self.get_anchor_feat_before = self.get_anchor_feat.data
        self.get_level_before = self.get_level.data
        self.get_offset_before = self.get_offset.data
        self.get_scaling_before = self.get_scaling.data

        self.get_extra_level_before = self.get_extra_level.data
        self.get_opacity_before = self.get_opacity.data

        # 关键:保存未激活的scaling和opacity
        self._lod_scaling_before = self._lod_scaling.data.clone()  # log空间
        self.lod_rotation_before = self.lod_rotation.data.clone()
        self.lod_opacity_before = self.lod_opacity.data.clone()   # inverse_sigmoid空间
        self.get_extra_level_before = self.get_extra_level.data.clone()
        
        # 2. 最快的MLP复制方式
        for name in ['unified_mlp']:
            source = getattr(self, f'unified_mlp')
            
            # 使用 deepcopy 但立即转为推理模式
            import copy
            target = copy.deepcopy(source)
            target.eval()
            
            # 关键：使用 @torch.jit.script 编译
            if torch.jit.is_scripting():
                target = torch.jit.script(target)
            
            for p in target.parameters():
                p.requires_grad_(False)
                # 可选：转为半精度加速
            
            setattr(self, f'unified_mlp_static', target)

    #@torch.inference_mode()
    def generate_neural_gaussians_prev(self, viewpoint_camera, visible_mask=None, is_train=None, grad=False):
        ## view frustum filtering for acceleration    
        if visible_mask is None:
            visible_mask = torch.ones(self.get_anchor_before.shape[0], dtype=torch.bool, device=self.get_anchor_before.device)
        
        @torch.inference_mode(mode=(not grad))
        def forward(self, viewpoint_camera, visible_mask=None, is_train=None, grad=False):

            anchors = self.get_anchor_before[visible_mask]
            feat = self.get_anchor_feat_before[visible_mask]
            grid_offsets = self.get_offset_before[visible_mask]
            grid_scaling = self.get_scaling_before[visible_mask]
            
            ## get view properties for anchor
            ob_view = anchors - viewpoint_camera.camera_center
            # dist
            ob_dist = ob_view.norm(dim=1, keepdim=True)
            # view
            ob_view = ob_view / ob_dist
            cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)
            with torch.no_grad():
                # MLP forward pass - 移除autocast，因为外层已经控制了梯度
                neural_opacity, scale_rot, color = self.unified_mlp_static(cat_local_view_wodist)
                
            # opacity mask generation
            neural_opacity = neural_opacity.reshape([-1, 1])
            mask = (neural_opacity > 0.0)
            mask = mask.view(-1)
            
            # select opacity 
            opacity = neural_opacity[mask]
            color = color.reshape([anchors.shape[0]*self.n_offsets, ((self.max_sh_degree + 1) ** 2)*3])
            scale_rot = scale_rot.reshape([anchors.shape[0]*self.n_offsets, 7])
            
            # offsets
            offsets = grid_offsets.view([-1, 3])
            
            # combine for parallel masking
            concatenated = torch.cat([grid_scaling, anchors], dim=-1)
            concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
            concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
            masked = concatenated_all[mask]
            dim_color = ((self.max_sh_degree + 1) ** 2)*3
            scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, dim_color, 7, 3], dim=-1)
            color_end = color.reshape([-1, ((self.max_sh_degree + 1) ** 2), 3])
            
            # post-process cov
            scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
            rot = self.rotation_activation(scale_rot[:, 3:7])
            
            # post-process offsets to get centers for gaussians
            offsets = offsets * scaling_repeat[:, :3]
            xyz = repeat_anchor + offsets

            # 保存结果时不需要再次detach（如果grad=True已经detach了，如果grad=False本来就没有梯度）
            if is_train:
                neural_scaling_repeat, _, _, neural_scale_rot, neural_offsets = concatenated_all.split([6, 3, dim_color, 7, 3], dim=-1)
                neural_scaling = neural_scaling_repeat[:, 3:] * torch.sigmoid(neural_scale_rot[:, :3])
                
                self.xyz_save = xyz
                self.color_all_save = color_end
                self.opacity_save = opacity
                self.scaling_save = scaling
                self.rot_save = rot
                self.neural_opacity_save = neural_opacity
                self.neural_scaling_save = neural_scaling
                self.mask_save = mask
            else:
                self.xyz_save = xyz
                self.color_all_save = color_end
                self.opacity_save = opacity
                self.scaling_save = scaling
                self.rot_save = rot
                self.mask_save = mask
                self.anchors_save = anchors

        return forward(self, viewpoint_camera, visible_mask, is_train, grad)
     

    def octree_sample(self, data, init_pos): #构建一个多分辨率体素层级结构，通过逐层缩小体素尺寸，对输入数据进行分层采样。
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda() 
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            #计算当前层的体素大小,实现多尺度体素划分，高层体素更细，低层体素更粗 
            new_positions = torch.unique(torch.round((data - init_pos) / cur_size), dim=0) * cur_size + init_pos #从原始数据中提取当前分辨率层的体素中心点坐标。
            #根据当前体素尺寸计算新点的位置,将坐标舍入到最近的整数格点，相当于将点映射到体素网格索引上。删除重复的网格点，只保留唯一的体素位置。将网格索引还原为世界坐标下的实际点位置。
            new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            #生成一个与 new_positions 数量相同的整型张量。每个元素的值为 cur_level。表示这些点属于第 cur_level 层。
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level = torch.concat((self._level, new_level), dim=0)
            #累积构建出所有层的体素点集和对应层级信息。
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def norm_decoders(self, att_name=None):
        for param_name in self.param_names:
            if att_name is not None and param_name!=att_name:
                continue
            decoder = self.latent_decoders[param_name]
            if not isinstance(decoder, DecoderIdentity) and decoder.norm!="none":
                decoder.normalize(self._latents[param_name])
                
    def create_from_pcd(self, pcd, point_cloud: BasicPointCloud, spatial_lr_scale : float, ignore_colors: bool = False):
        self.spatial_lr_scale = spatial_lr_scale

        box_min = torch.min(pcd)*self.extend
        box_max = torch.max(pcd)*self.extend
        box_d = box_max - box_min
        #print(box_d)  #3.0886

        if self.base_layer < 0:
            default_voxel_size = 0.02
            self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda() 
        #print(self.init_pos) #tensor([-1.3551, -1.3551, -1.3551]
        self.octree_sample(pcd, self.init_pos)

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)

        print(f'Branches of Tree: {self.fork}')
        print(f'Base Layer of Tree: {self.base_layer}')
        print(f'Visible Threshold: {self.visible_threshold}')
        print(f'Appearance Embedding Dimension: {self.appearance_dim}') 
        print(f'LOD Levels: {self.levels}')
        print(f'Size Levels: {self._level.size()}')
        print(f'Initial Levels: {self.init_level}')
        print(f'Initial Voxel Number: {self.positions.shape[0]}')
        print(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        print(f'Max Voxel Size: {self.voxel_size}')
        print(f'positions Size: {self.positions.shape[0]}')
        offsets = torch.zeros((self.positions.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((self.positions.shape[0], self.feat_dim)).float().cuda()
        dist2 = torch.clamp_min(distCUDA2(self.positions).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        scales = torch.clamp(scales, -10, 4) #torch.clamp 函数将 scales 张量中的所有元素值限制在 [-10, 4] 这个区间内。

        rots = torch.zeros((self.positions.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((self.positions.shape[0], 1), dtype=torch.float, device="cuda"))

        self.lod_rotation = nn.Parameter(rots.requires_grad_(False))
        self.lod_opacity = nn.Parameter(opacities.requires_grad_(False))
        self._level = self._level.unsqueeze(dim=1)
        flow = torch.zeros_like(self.positions)

        self._extra_level = torch.zeros(self.positions.shape[0], dtype=torch.float, device="cuda")
        self._anchor_mask = torch.ones(self.positions.shape[0], dtype=torch.bool, device="cuda")
        self._anchor_mask_grad = torch.zeros(self.positions.shape[0], dtype=torch.bool, device="cuda")
        self._anchor_mask_grad_report = torch.ones(self.positions.shape[0], dtype=torch.bool, device="cuda")

        #print("Number of points at _anchor : ",self._anchor.size())       
        # ###################################### Init latents ##########################################
        self._latents = OrderedDict([(n,None) for n in self.param_names])

        fused_point_cloud = torch.tensor(np.asarray(self.positions.cpu())).float().cuda()
        init = self.latent_decoders["anchor"].invert(fused_point_cloud)
        self._latents["anchor"] = nn.Parameter(init.requires_grad_(True))

        init = anchors_feat[:,:].contiguous()
        if isinstance(self.latent_decoders["f_feature_anchor"], LatentDecoder):
            init = torch.zeros((anchors_feat.size(0),self.latent_decoders["f_feature_anchor"].latent_dim)).to(init).contiguous()
        self._latents["f_feature_anchor"] = nn.Parameter(init.requires_grad_(True))

        if not isinstance(self.latent_decoders["flow"], DecoderIdentity):
            self._latents["flow"] = nn.Parameter(torch.zeros(flow.shape[0],self.latent_decoders["flow"].latent_dim), 
                                                 requires_grad=True, device=flow.device)
        else:
            self._latents["flow"] = nn.Parameter(flow.requires_grad_(True)) 
        
        if self.latent_args.sc_invert_type == "autoenc" and not isinstance(self.latent_decoders["sc"], DecoderIdentity):
            init, decoder_state_dict = init_latents(self.latent_args, scales,"sc", lambda_distortion=0.0)
            self.latent_decoders["sc"].load_state_dict(decoder_state_dict)
        else:
            init = self.latent_decoders["sc"].invert(scales)
        self._latents["sc"] = nn.Parameter(init.requires_grad_(True))

        init = offsets.transpose(1, 2).contiguous()
        if isinstance(self.latent_decoders["offset"], LatentDecoder):
            init = torch.zeros((offsets.size(0),self.latent_decoders["offset"].latent_dim)).to(init).contiguous()
        self._latents["offset"] = nn.Parameter(init.requires_grad_(True))

        #将一个名为 flow 的张量通过 requires_grad_ 设置为可求导，并将其作为一个新的参数存储在 self._latents["flow"] 中 
        ##########################################################################################

    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level

    '''@torch.no_grad()
    def update_points_flow(self):
        if isinstance(self.latent_decoders["anchor"], DecoderIdentity) \
            or isinstance(self.latent_decoders["anchor"], LatentDecoder):
            new_anchor = self._anchor+self._flow 
            self._latents["anchor"].data = self.latent_decoders["anchor"].invert(new_anchor)

        self._latents["flow"] *= 0'''


    def weed_out(self, anchor_positions, anchor_levels): #过滤掉那些在多个相机中不可见或者视图不够清晰的锚点（anchor）
        visible_count = torch.zeros(anchor_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((anchor_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (anchor_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask

    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        #print((self.get_anchor))
        anchor_pos = self.latent_decoders["anchor"](self._latents["anchor"])+ (self.voxel_size/2) / (float(self.fork) ** self._level)
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        
        is_training = self.get_unified_mlp.training
        if self.progressive and is_training:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    def set_anchor_mask_grad(self, cur_level=1,frame_idx=1):
        cond_mask = (self._grad_level == cur_level)
        if frame_idx>2:
            old_mask = self._anchor_mask_grad.clone()
            self._anchor_mask_grad |= cond_mask   #|
            self.diff_mask = (~old_mask) & cond_mask
        else:
            self._anchor_mask_grad = cond_mask   #|
            self.diff_mask = None   #|
    

    def set_anchor_mask_level(self, cur_level):
        self._anchor_mask_level = (self._level.squeeze(dim=1) == cur_level)

    def set_anchor_mask_grad_level(self, cur_level):
        self._anchor_mask_grad_report = (self._grad_level == cur_level)

    def eval(self):
        self.unified_mlp.eval()    

    def train(self):
        self.unified_mlp.train() 
 
    def size_mlp(self,model):
        total_size_bits = 0
        # 遍历 model.parameters() 获取每个参数的大小
        for param in model.parameters():
            total_size_bits += param.numel() * torch.finfo(param.dtype).bits
        return total_size_bits

    def size(self):
        """Calculate compressed model size in bits for storage estimation."""
        with torch.no_grad():
            ldec_size = latents_size = 0
            ldec_size += self.size_mlp(self.unified_mlp) * 2

            for param_name in self.param_names:
                if param_name == "flow":
                    continue
                    
                # Add decoder size
                ldec_size += self.latent_decoders[param_name].size()
                decoder = self.latent_decoders[param_name]
                
                # Calculate latent storage size based on decoder type
                if isinstance(decoder, DecoderIdentity) \
                    or (type(decoder)==LatentDecoderRes and decoder.identity):
                    p = self._latents[param_name]
                    latents_size += p.numel()*torch.finfo(p.dtype).bits#*mask

            level_size_raw = self._level.numel() * self._level.element_size() * 8  # element_size() 
            compression_ratio = 0.3 
            latents_size = latents_size + level_size_raw* compression_ratio

        return ldec_size + latents_size
 
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.anchor_gradient_accum = torch.zeros((self.positions.shape[0], 1), device="cuda")
        self.infl_accum = torch.zeros((self.positions.shape[0]), device="cuda")
        self.denom = torch.zeros((self.positions.shape[0], 1), device="cuda")
        self.denom_lod = torch.zeros((self.positions.shape[0], 1), device="cuda")
        self.infl_denom = torch.zeros((self.positions.shape[0]), device="cuda")
        self.added_mask = None

        self.opacity_accum = torch.zeros((self.positions.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.positions.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_gradient_accum_lod = torch.zeros((self.positions.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.positions.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom_lod = torch.zeros((self.positions.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.positions.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self.lod_opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.lod_rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.unified_mlp.parameters(), 'lr': training_args.mlp_unified_lr_init, "name": "unified_mlp"},
        ]
        self.optimizer2 = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.lr_scaling = OrderedDict()
        for i,param in enumerate(self.param_names):
            decoder = self.latent_decoders[param]
            if type(decoder) == DecoderIdentity or (type(decoder)== LatentDecoderRes and decoder.identity):
                self.lr_scaling[param] = 1.0
            else:
                self.lr_scaling[param] = training_args.latents_lr_scaling[i]

        self.orig_lr = OrderedDict({"anchor":training_args.position_lr_init, 
                                    "f_feature_anchor":training_args.features_dc_lr, 
                                     "flow":training_args.flow_lr,
                                     "sc":training_args.scaling_lr,
                                      "offset":training_args.offset_lr_init})
        lr = {
                'anchor':training_args.position_lr_init * self.spatial_lr_scale*self.lr_scaling["anchor"],
                'f_feature_anchor':training_args.features_dc_lr*self.lr_scaling["f_feature_anchor"],
                'flow':training_args.flow_lr*self.lr_scaling["flow"],
                'sc':training_args.scaling_lr*self.lr_scaling["sc"],
                "offset":training_args.offset_lr_init*self.spatial_lr_scale,
            }
        l = []
        for i,param in enumerate(self.param_names):
            l += [{'params': [self._latents[param]], 'lr': lr[param], "name": param}]
            if not isinstance(self.latent_decoders[param], DecoderIdentity):
                l += [{'params': self.latent_decoders[param].parameters(), 'lr': training_args.ldecs_lr[i], "name":f"ldec_{param}"}]

        self.optimizer1 = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.unified_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_unified_lr_init,
                                                    lr_final=training_args.mlp_unified_lr_final,
                                                    lr_delay_mult=training_args.mlp_unified_lr_delay_mult,
                                                    max_steps=training_args.mlp_unified_lr_max_steps)
                         

    def training_lodgrad(self):
        self.offset_gradient_accum_lod = torch.zeros((self.get_anchor_before.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom_lod = torch.zeros((self.get_anchor_before.shape[0]*self.n_offsets, 1), device="cuda")

    def training_setup_n(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.anchor_gradient_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.infl_accum = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        self.denom = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.denom_lod = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.infl_denom = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        self.added_mask = None

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        l = [
            {'params': [self.lod_opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.lod_rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': self.unified_mlp.parameters(), 'lr': training_args.mlp_unified_lr_init, "name": "unified_mlp"},
        ]
        self.optimizer2 = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.lr_scaling = OrderedDict()
        for i,param in enumerate(self.param_names):
            decoder = self.latent_decoders[param]
            if type(decoder) == DecoderIdentity or (type(decoder)== LatentDecoderRes and decoder.identity):
                self.lr_scaling[param] = 1.0
            else:
                self.lr_scaling[param] = training_args.latents_lr_scaling[i]

        self.orig_lr = OrderedDict({"anchor":training_args.position_lr_init, 
                                    "f_feature_anchor":training_args.features_dc_lr, 
                                     "flow":training_args.flow_lr,
                                      "sc":training_args.scaling_lr,
                                      "offset":training_args.offset_lr_init})
        lr = {
                'anchor':training_args.position_lr_init * self.spatial_lr_scale*self.lr_scaling["anchor"],
                'f_feature_anchor':training_args.features_dc_lr*self.lr_scaling["f_feature_anchor"],
                'flow':training_args.flow_lr*self.lr_scaling["flow"],
                'sc':training_args.scaling_lr*self.lr_scaling["sc"],
                "offset":training_args.offset_lr_init*self.spatial_lr_scale*self.lr_scaling["offset"],
            }
        l = []
        for i,param in enumerate(self.param_names):
            l += [{'params': [self._latents[param]], 'lr': lr[param], "name": param}]
            if not isinstance(self.latent_decoders[param], DecoderIdentity):
                l += [{'params': self.latent_decoders[param].parameters(), 'lr': training_args.ldecs_lr[i], "name":f"ldec_{param}"}]

        self.optimizer1 = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
 
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.unified_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_unified_lr_init,
                                                    lr_final=training_args.mlp_unified_lr_final,
                                                    lr_delay_mult=training_args.mlp_unified_lr_delay_mult,
                                                    max_steps=training_args.mlp_unified_lr_max_steps)

 
 
    def update_grads(self):
        frz, atts, masks = self.get_frz, self.get_atts, self.get_masks
        for att_name in atts:
            if frz[att_name] == 'st':
                mask = masks[att_name] if frz[att_name] == 'st' else ~masks[att_name]
                if "f_" in att_name:
                    atts[att_name].grad *= mask.unsqueeze(-1)
                else:
                    atts[att_name].grad *= mask

    def update_learning_rate(self, iteration, latent_args: QuantizeParams):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer2.param_groups:

            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "unified_mlp":
                lr = self.unified_mlp_scheduler_args(iteration)
                param_group['lr'] = lr
 

        for param_group in self.optimizer1.param_groups:
            if param_group["name"] == "anchor":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr*self.lr_scaling["anchor"]
            elif param_group["name"] in self.param_names:
                idx = self.param_names.index(param_group["name"])
                if latent_args.latent_scale_norm[idx] == "div":
                    lr = self.orig_lr[param_group["name"]]*self.lr_scaling[param_group["name"]]
                    lr /= self.latent_decoders[param_group["name"]].scale_norm()
                    param_group['lr'] = lr
 
    def construct_list_of_attributes_lod(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        l.append('level')
        l.append('extra_level')
        l.append('info')
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._lod_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.lod_rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply_lod(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
        infos = np.zeros_like(levels, dtype=np.float32)
        infos[0, 0] = self.voxel_size
        infos[1, 0] = self.standard_dist

        anchor_feats = self._anchor_feat.detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.lod_opacity.detach().cpu().numpy()
        scales = self._lod_scaling.detach().cpu().numpy()
        rots = self.lod_rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_lod()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, infos, offsets, anchor_feats, opacities, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply(self, path, mask):
        mkdir_p(os.path.dirname(path))

        if mask is not None:
            xyz = self._anchor[mask].detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity[mask].detach().cpu().numpy()
            scale = self._lod_scaling[mask].detach().cpu().numpy()
            rotation = self._rotation[mask].detach().cpu().numpy()
        else:
            xyz = self._anchor.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._lod_scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

        vertex_ids = np.arange(xyz.shape[0])
        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
        dtype_full.extend([(attribute, 'f4') for attribute in 
                        [f'f_dc_{i}' for i in range(f_dc.shape[1])]])
        dtype_full.extend([(attribute, 'f4') for attribute in 
                        [f'f_rest_{i}' for i in range(f_rest.shape[1])]])
        dtype_full.extend([('opacity', 'f4')])
        dtype_full.extend([(attribute, 'f4') for attribute in 
                        [f'scale_{i}' for i in range(scale.shape[1])]])
        dtype_full.extend([(attribute, 'f4') for attribute in 
                        [f'rot_{i}' for i in range(rotation.shape[1])]])
        dtype_full.append(('vertex_id', 'i4'))  # Add the vertex_id field

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
        for i in range(f_dc.shape[1]):
            elements[f'f_dc_{i}'] = f_dc[:, i]
        for i in range(f_rest.shape[1]):
            elements[f'f_rest_{i}'] = f_rest[:, i]
        elements['opacity'] = opacities[:, 0]  # Fix the shape here
        for i in range(scale.shape[1]):
            elements[f'scale_{i}'] = scale[:, i]
        for i in range(rotation.shape[1]):
            elements[f'rot_{i}'] = rotation[:, i]
        elements['vertex_id'] = vertex_ids

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
        infos = np.zeros_like(levels, dtype=np.float32)
        infos[0, 0] = self.voxel_size
        infos[1, 0] = self.standard_dist

        anchor_feats = self._anchor_feat.detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.lod_opacity.detach().cpu().numpy()
        scales = self._lod_scaling.detach().cpu().numpy()
        rots = self.lod_rotation.detach().cpu().numpy()
         
        # 获取路径的目录和文件名部分
        dirname, filename = os.path.split(path)
        # 获取文件名的主干和扩展名部分
        name, ext = os.path.splitext(filename)
        # 在文件名的主干后添加 .lod
        new_filename = name + 'lod' + ext
        # 合成新的完整路径
        new_path = os.path.join(dirname, new_filename)
        mkdir_p(os.path.dirname(new_path))
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_lod()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, infos, offsets, anchor_feats, opacities, scales, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(new_path)

    def plot_levels(self):
        for level in range(self.levels):
            level_mask = (self._level == level).squeeze(dim=1)
            print(f'Level {level}: {torch.sum(level_mask).item()}, Ratio: {torch.sum(level_mask).item()/self._level.shape[0]}')
 

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        if type(self.latent_decoders["op"]) == LatentDecoderRes:
            opacities_new = self.latent_decoders["op"].invert(opacities_new-self.latent_decoders["op"].decoded_att)
        else:
            opacities_new = self.latent_decoders["op"].invert(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "op")
        self._latents["op"] = optimizable_tensors["op"]
 
    def replace_tensor_to_optimizer(self, tensor, name, lr=None):
        optimizable_tensors = {}
        assert "ldec" not in name, "Latent decoder params cannot be replaced!"
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                if lr is not None:
                    group['lr'] = lr
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "ldec" in group["name"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
                    
    def cat_tensors_to_optimizer(self, tensors_dict1, tensors_dict2):
        optimizable_tensors = {}
        #print(tensors_dict.keys())
        for group in self.optimizer2.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict1[group["name"]]
            stored_state = self.optimizer2.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer2.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer2.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        optimizable_tensors2 = {}
        #masks = self.get_masks
        for group in self.optimizer1.param_groups:
            if "ldec" in group["name"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict2[group["name"]]
            stored_state = self.optimizer1.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer1.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer1.state[group['params'][0]] = stored_state

                optimizable_tensors2[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors2[group["name"]] = group["params"][0]
   
        return optimizable_tensors, optimizable_tensors2
 
    # statis grad information to guide liftting. 
    def training_statis_lod(self, viewspace_point_tensor, update_filter, offset_selection_mask=None,anchor_visible_mask=None):
        # update neural gaussian statis
        combined_mask = torch.zeros_like(self.offset_gradient_accum_lod, dtype=torch.bool).squeeze(dim=1)
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask[anchor_visible_mask] = offset_selection_mask

        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)

        # 基于平均梯度决定是否需要致密化
        self.offset_gradient_accum_lod[combined_mask] += grad_norm
        self.offset_denom_lod[combined_mask] += 1 


    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_mask=None, anchor_visible_mask=None):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        if anchor_mask is None:
            anchor_mask = torch.ones(self.opacity_accum.shape[0], dtype=torch.bool, device = self.opacity_accum.device)

        self.opacity_accum[anchor_mask] += temp_opacity.sum(dim=1, keepdim=True)
        # update anchor visiting statis
        self.anchor_demon[anchor_mask] += 1
        # update neural gaussian statis

        anchor_mask = anchor_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # 基于平均梯度决定是否需要致密化
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def training_statis_n(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask):
        # update opacity stats anchor_visible_mask.sum()
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum += temp_opacity.sum(dim=1, keepdim=True)
        # update anchor visiting statis
        self.anchor_demon += 1
        # update neural gaussian statis
        # anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask = offset_selection_mask 
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter[:offset_selection_mask.sum()] 

        #combined_mask = temp_mask & update_filter[:offset_selection_mask.shape[0]]  
        grad_norm = torch.norm(
            viewspace_point_tensor.grad[:offset_selection_mask.sum(), :2], 
            dim=-1, 
            keepdim=True
        )

        # 基于平均梯度决定是否需要致密化
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1
        
    def _prune_anchor_optimizer(self, mask):

        optimizable_tensors1 = {}
        for group in self.optimizer2.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer2.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer2.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer2.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    #temp = torch.clamp(temp, max=0.05)
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors1[group["name"]] = group["params"][0]
            else:
                # 没有优化状态：只更新参数
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp = torch.clamp(temp, max=0.05)
                    with torch.no_grad():
                        group["params"][0][:, 3:] = temp
                    #group["params"][0][:,3:] = temp
                optimizable_tensors1[group["name"]] = group["params"][0]
        optimizable_tensors2 = {}
        for group in self.optimizer1.param_groups:
            if "ldec" in group["name"]:
                continue
            stored_state = self.optimizer1.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer1.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer1.state[group['params'][0]] = stored_state

                optimizable_tensors2[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors2[group["name"]] = group["params"][0]

        return optimizable_tensors1, optimizable_tensors2

    def add_anchor(self):
        new_anchor0 = self.get_anchor_before[self.diff_mask] #torch.zeros([0, 3], dtype=torch.float, device='cuda')
        cur_size = self.voxel_size / (float(self.fork) ** self.get_level_before[self.diff_mask]) #** ((self.get_level_before))

        new_anchor = torch.tensor(np.asarray(new_anchor0.cpu())).float().cuda()
        #init = self.latent_decoders["anchor"].invert(fused_anchor)
        #new_anchor = nn.Parameter(fused_anchor.requires_grad_(True))

        init_anchor_feat = self.get_anchor_feat_before[self.diff_mask].contiguous()
        #new_feat = torch.zeros((new_anchor0.size(0), self.latent_decoders["f_feature_anchor"].latent_dim)).to(init_anchor_feat).contiguous()

        new_feat = torch.zeros([new_anchor0.size(0), self.latent_decoders["f_feature_anchor"].latent_dim], dtype=torch.float, device='cuda')

        #new_feat = nn.Parameter(init.requires_grad_(True))

        new_offsets0 = torch.zeros((new_anchor0.shape[0], 3, self.n_offsets)).float().cuda()
        if isinstance(self.latent_decoders["offset"], LatentDecoder):
            new_offsets = torch.zeros((new_offsets0.size(0),
                                    self.latent_decoders["offset"].latent_dim)).contiguous().cuda()
        else:
            new_offsets = new_offsets0
        
        new_offsets = nn.Parameter(new_offsets.requires_grad_(True))

        new_scaling0 = torch.ones_like(new_anchor0).repeat([1,2]).float().cuda()*cur_size #*cur_size # *0.05
        new_scaling = torch.log(new_scaling0)
        #new_scaling = torch.clamp(new_scaling, -10, 4)
        #init = self.latent_decoders["sc"].invert(new_scaling)
        #new_scaling = nn.Parameter(new_scaling.requires_grad_(True))
        new_level = self.get_level_before[self.diff_mask]

        # 关键修改:使用原始空间的参数
        #new_scaling = self._lod_scaling_before[self.diff_mask]  # 已经在log空间
        new_opacities = inverse_sigmoid(0.05 * torch.ones((new_anchor0.shape[0], 1), dtype=torch.float, device="cuda"))
        new_rotation = torch.zeros([new_anchor0.shape[0], 4], dtype=torch.float, device='cuda')
        new_rotation[:,0] = 1.0
        new_extra_level = torch.zeros(new_anchor0.shape[0], dtype=torch.float, device='cuda')

        #new_rotation = self.lod_rotation_before[self.diff_mask]
        #new_opacities = self.lod_opacity_before[self.diff_mask]  # 已经在inverse_sigmoid空间
        #new_extra_level = self.get_extra_level_before[self.diff_mask]
        new_flow = torch.zeros_like(new_anchor0) #torch.zeros(new_anchor0.shape[0])

        #将新锚点添加到优化器
        d1 = {
            "rotation": new_rotation,
            "opacity": new_opacities,
        }   
        d2 = {
            "anchor": new_anchor,
            "sc": new_scaling,
            "f_feature_anchor": new_feat,
            "flow": new_flow,
            "offset": new_offsets,
        }   

        torch.cuda.empty_cache()
        optimizable_tensors, optimizable_tensors2 = self.cat_tensors_to_optimizer(d1,d2)
        self.lod_rotation = optimizable_tensors["rotation"]
        self.lod_opacity = optimizable_tensors["opacity"]
        for param in self.param_names:
            self._latents[param] = optimizable_tensors2[param]

        self._level = torch.cat([self._level, new_level], dim=0)
        self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)

        new_offset_gradient_accum = torch.zeros((self.diff_mask.sum()*self.n_offsets, 1), device="cuda").float()
        new_offset_denom = torch.zeros((self.diff_mask.sum()*self.n_offsets, 1), device="cuda").float()
        new_anchor_demon = torch.zeros((self.diff_mask.sum(), 1), device="cuda").float()
        new_opacity_accum = torch.zeros((self.diff_mask.sum(), 1), device="cuda").float()
        new_anchor_gradient_accum = torch.zeros((self.diff_mask.sum(), 1), device="cuda").float()
        new_denom = torch.zeros((self.diff_mask.sum(), 1), device="cuda").float()
        new_denom_lod = torch.zeros((self.diff_mask.sum(), 1), device="cuda").float()

        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, new_offset_gradient_accum], dim=0)
        self.offset_denom = torch.cat([self.offset_denom, new_offset_denom], dim=0)
        self.anchor_demon = torch.cat([self.anchor_demon, new_anchor_demon], dim=0)
        self.opacity_accum = torch.cat([self.opacity_accum, new_opacity_accum], dim=0)
        self.anchor_gradient_accum = torch.cat([self.anchor_gradient_accum, new_anchor_gradient_accum], dim=0)
        self.denom = torch.cat([self.denom, new_denom], dim=0)
        self.denom_lod = torch.cat([self.denom_lod, new_denom_lod], dim=0) 

        # ===== 新增：同步更新 LatentDecoderRes 的 decoded_att =====
        for param_name in ["anchor", "f_feature_anchor", "sc", "flow", "offset"]:
            decoder = self.latent_decoders[param_name]
            if isinstance(decoder, LatentDecoderRes) and hasattr(decoder, 'decoded_att'):
                # 为新增的锚点创建对应的 decoded_att
                if param_name == "anchor":
                    new_decoded = new_anchor0  # 使用新锚点的坐标
                elif param_name == "f_feature_anchor":
                    new_decoded = torch.zeros((new_anchor0.shape[0], self.feat_dim), dtype=torch.float, device='cuda') #self.get_anchor_feat_before[self.diff_mask].contiguous()
                elif param_name == "sc":
                    new_decoded = new_scaling #self.get_scaling_before[self.diff_mask].contiguous()  # 使用新的 new_scaling0
                elif param_name == "flow":
                    new_decoded = torch.zeros_like(new_anchor0)
                elif param_name == "offset":
                    new_decoded = torch.zeros((new_anchor0.shape[0], 30), dtype=torch.float, device='cuda')  #self.get_offset_before[self.diff_mask].reshape(new_anchor0.shape[0], 3*self.n_offsets).contiguous() 
                
                decoder.decoded_att = torch.cat([decoder.decoded_att, new_decoded], dim=0)


    def refine_anchor(self):
        """根据 _anchor_mask_grad 筛选锚点，保留 mask 为 True 的点"""
        # 步骤1: 使用 _prune_anchor_optimizer 同步更新优化器
        optimizable_tensors1, optimizable_tensors2 = self._prune_anchor_optimizer(self._anchor_mask_grad)
        torch.cuda.empty_cache()

        if not self._anchor_mask_grad.any():
            print(f"Warning: _anchor_mask_grad is all False! No anchors will be retained.")
            return
    
        # 步骤2: 从优化器返回的字典中获取更新后的可学习参数
        self._latents["anchor"] = optimizable_tensors2["anchor"]
        self._latents["offset"]  = optimizable_tensors2["offset"]
        
        self._latents["sc"] = optimizable_tensors2["sc"]
        self._latents["f_feature_anchor"] = optimizable_tensors2["f_feature_anchor"]
        self._latents["flow"] = optimizable_tensors2["flow"]
        self.lod_opacity = optimizable_tensors1["opacity"]
        self.lod_rotation = optimizable_tensors1["rotation"]

        # 注意：lod_opacity 和 lod_rotation 虽然也在优化器中，但这里可能不需要更新
        # 如果需要，也要从 optimizable_tensors 中获取
        # 步骤3: 对于不在优化器中的张量，直接使用索引操作
        self._level = self._level[self._anchor_mask_grad]
        self._extra_level = self._extra_level[self._anchor_mask_grad]

        # ===== 新增：同步更新 LatentDecoderRes 的 decoded_att =====
        for param_name in ["anchor", "f_feature_anchor", "sc", "flow", "offset"]:
            decoder = self.latent_decoders[param_name]
            if isinstance(decoder, LatentDecoderRes):
                # 用筛选后的 decoded_att 更新
                decoder.decoded_att = decoder.decoded_att[self._anchor_mask_grad]

        self.offset_gradient_accum = torch.zeros((self._anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self._anchor.shape[0]*self.n_offsets, 1), device="cuda")
        # 步骤4: 同步更新其他相关的统计张量
        # 这些张量在 training_setup 中初始化，但不是 nn.Parameter
        if hasattr(self, 'opacity_accum') and self.opacity_accum.shape[0] > 0:
            self.opacity_accum = self.opacity_accum[self._anchor_mask_grad]
        
        if hasattr(self, 'anchor_demon') and self.anchor_demon.shape[0] > 0:
            self.anchor_demon = self.anchor_demon[self._anchor_mask_grad]
        
        if hasattr(self, 'anchor_gradient_accum') and self.anchor_gradient_accum.shape[0] > 0:
            self.anchor_gradient_accum = self.anchor_gradient_accum[self._anchor_mask_grad]
        
        if hasattr(self, 'denom') and self.denom.shape[0] > 0:
            self.denom = self.denom[self._anchor_mask_grad]
            
        if hasattr(self, 'denom_lod') and self.denom_lod.shape[0] > 0:
            self.denom_lod = self.denom_lod[self._anchor_mask_grad] 

    def prune_anchor(self,mask,frame_idx=0): #锚点剪枝
        valid_points_mask = ~mask
        #调用优化器剪枝，确保优化器与模型参数同步
        optimizable_tensors1, optimizable_tensors2 = self._prune_anchor_optimizer(valid_points_mask)
        torch.cuda.empty_cache()

        self._latents["anchor"] = optimizable_tensors2["anchor"]
        self._latents["offset"] = optimizable_tensors2["offset"]
        self.lod_opacity = optimizable_tensors1["opacity"]
        self.lod_rotation = optimizable_tensors1["rotation"]
        self._latents["sc"] = optimizable_tensors2["sc"]
        self._latents["f_feature_anchor"] = optimizable_tensors2["f_feature_anchor"]
        self._latents["flow"] = optimizable_tensors2["flow"]

        #同步更新所有与锚点相关的张量
        self._level = self._level[valid_points_mask]    
        self._extra_level = self._extra_level[valid_points_mask]

        if frame_idx ==1:
            self._anchor_mask_grad = self._anchor_mask_grad[valid_points_mask]
            self._anchor_mask_grad_report = self._anchor_mask_grad_report[valid_points_mask]

        # 注意：lod_opacity 和 lod_rotation 虽然也在优化器中，但这里可能不需要更新
        # 如果需要，也要从 optimizable_tensors 中获取
        # 步骤3: 对于不在优化器中的张量，直接使用索引操作
 
        # ===== 新增：同步更新 LatentDecoderRes 的 decoded_att =====
        for param_name in ["anchor", "f_feature_anchor", "sc", "flow", "offset"]:
            decoder = self.latent_decoders[param_name]
            if isinstance(decoder, LatentDecoderRes):
                # 用筛选后的 decoded_att 更新
                decoder.decoded_att = decoder.decoded_att[valid_points_mask]

        self.offset_gradient_accum = torch.zeros((self._anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self._anchor.shape[0]*self.n_offsets, 1), device="cuda")
        # 步骤4: 同步更新其他相关的统计张量
        # 这些张量在 training_setup 中初始化，但不是 nn.Parameter
 
 
 
    def adjust_anchor(self, iteration, check_interval=10, success_threshold=0.6, grad_threshold=0.0002, update_ratio=0.5, extra_ratio=0.25, extra_up=0.25, min_opacity=0.005,frame_idx=1):
        # # adding anchors
        # 计算每个偏移点的平均梯度强度 
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1] ## 计算平均梯度
        grads[grads.isnan()] = 0.0  # 处理除零导致的NaN
        grads_norm = torch.norm(grads, dim=-1)  # 计算梯度范数 
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1) # 筛选统计可靠的偏移点（访问次数足够）

        # 在高梯度区域添加新锚点
        self.anchor_growing(iteration, grads_norm, grad_threshold, update_ratio, extra_ratio, extra_up, offset_mask,frame_idx=frame_idx)
        
        # update offset_denom # 重置已处理的偏移点统计 为新增锚点分配统计空间
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors 平均不透明度低于阈值 已被充分评估（访问次数足够）
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom # 保留未被剪枝的统计信息 
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom  # 显式删除旧张量 更新统计张量，移除被剪枝锚点的数据
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            #prune_filter = torch.cat((prune_mask, torch.zeros(selected_pts_clone, device="cuda", dtype=bool)))
            self.prune_anchor(prune_mask,frame_idx=frame_idx)
 
    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):# 高效检测网格坐标重复，避免在同一位置创建多个锚点 分块处理防止大规模张量操作导致内存溢出
        # 分块处理，避免内存爆炸
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates

    def expand_grad_by_neighbors(self, xyz_save, level_val):
        all_same = self._grad_level * (self.get_level_before == level_val).squeeze()
        high_grad_indices = torch.where(all_same == 1)[0]
        if len(high_grad_indices) == 0:
            return
        # 一次性转到CPU和numpy
        xyz_np = xyz_save.detach().cpu().numpy()
        high_grad_indices_np = high_grad_indices.cpu().numpy()
        
        # 构建KD-Tree
        tree = KDTree(xyz_np)
        high_grad_points = xyz_np[high_grad_indices_np]
        
        # 定义搜索半径
        radius = float(self.voxel_size / 1.0)
        # 半径查询
        neighbor_indices_list = tree.query_ball_point(high_grad_points, r=radius)
        # 过滤空列表并合并（一步完成）
        non_empty_neighbors = [neighbors for neighbors in neighbor_indices_list if len(neighbors) > 0]
        
        if not non_empty_neighbors:
            return
        # 合并并去重
        all_neighbors = np.unique(np.concatenate(non_empty_neighbors))
        
        # 转回tensor并赋值
        device = self._grad_level.device
        neighbor_tensor = torch.from_numpy(all_neighbors).long().to(device)
        self._grad_level[neighbor_tensor] = 1
    

    def expand_grad_by_neighbors_all(self, xyz_save):
        high_grad_indices = torch.where(self._grad_level == 1)[0]
        if len(high_grad_indices) == 0:
            return
        # 一次性转到CPU和numpy
        xyz_np = xyz_save.detach().cpu().numpy()
        high_grad_indices_np = high_grad_indices.cpu().numpy()
        
        # 构建KD-Tree
        tree = KDTree(xyz_np)
        high_grad_points = xyz_np[high_grad_indices_np]
        
        # 定义搜索半径
        radius = float(self.voxel_size / 0.12)
        # 半径查询
        neighbor_indices_list = tree.query_ball_point(high_grad_points, r=radius)
        # 过滤空列表并合并（一步完成）
        non_empty_neighbors = [neighbors for neighbors in neighbor_indices_list if len(neighbors) > 0]
        if not non_empty_neighbors:
            return
        # 合并并去重
        all_neighbors = np.unique(np.concatenate(non_empty_neighbors))
        # 转回tensor并赋值
        device = self._grad_level.device
        neighbor_tensor = torch.from_numpy(all_neighbors).long().to(device)
        self._grad_level[neighbor_tensor] = 1


    def replace_tensor_to_optimizer(self, tensor, name, lr=None):
        optimizable_tensors = {}
        assert "ldec" not in name, "Latent decoder params cannot be replaced!"
        for group in self.optimizer1.param_groups:
            if group["name"] == name:
                if lr is not None:
                    group['lr'] = lr
                stored_state = self.optimizer1.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer1.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer1.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors



    def update_residuals(self):
        """Initialize residual encoders for temporal compression in subsequent frames."""
        atts = self.get_atts # All the latent variables
        
        # Initialize parent mapping for tracking Gaussian relationships        
        for i, att_name in enumerate(atts):
            if self.latent_args.quant_type[i] == 'sq_res':
                decoded_att = self.latent_decoders[att_name](self._latents[att_name])
                
                if self.frame_idx == 2:
                    # Switch from identity to residual decoder after frame 1
                    assert isinstance(self.latent_decoders[att_name], DecoderIdentity)
                    decoder = LatentDecoderRes(
                        latent_dim=self.latent_args.latent_dim[i],
                        feature_dim=self.feature_dims[att_name],
                        ldecode_matrix=self.latent_args.ldecode_matrix[i],
                        latent_norm=self.latent_args.latent_norm[i],
                        num_layers_dec=self.latent_args.num_layers_dec[i],
                        hidden_dim_dec=self.latent_args.hidden_dim_dec[i],
                        activation=self.latent_args.activation[i],
                        use_shift=self.latent_args.use_shift[i],
                        ldec_std=self.latent_args.ldec_std[i],
                        final_activation=self.latent_args.final_activation[i],
                    ).cuda()
                    self.latent_decoders[att_name] = decoder
                else:
                    # Verify residual decoder type for subsequent frames
                    decoder = self.latent_decoders[att_name]
                    assert isinstance(self.latent_decoders[att_name], LatentDecoderRes)

                self.latent_decoders[att_name].frame_idx = self.frame_idx

                # Initialize decoder with previous frame's decoded attributes
                if ("offset" in att_name) and self.frame_idx == 2: #"f_" in att_name or 
                    decoder.init_decoded(decoded_att.reshape(decoded_att.shape[0], -1))  # SH coefficients in (N, X)
                else:
                    decoder.init_decoded(decoded_att)

                # Set up new latent parameters for current frame
                if type(self.latent_decoders[att_name])== LatentDecoderRes \
                    and self.latent_args.quant_after[i]>0.0:
                    # Start in identity mode, switch to quantized later
                    self.latent_decoders[att_name].identity = True
                    decoder = self.latent_decoders[att_name]
                    latent = torch.zeros_like(decoder.decoded_att)
                    self._latents[att_name] = nn.Parameter(latent.requires_grad_(True))
                else:
                    # Initialize with zero residuals
                    self._latents[att_name] = nn.Parameter(
                                                torch.zeros((self._latents[att_name].shape[0],
                                                                self.latent_args.latent_dim[i]), 
                                                            dtype=torch.float, 
                                                            device="cuda").requires_grad_(True)
                                                )
            

    def anchor_grad_2frame_threthold(self):
        """
        Args:
            grads: 梯度
            num_levels: grad_level的数量（固定为3）
            level_ratios: 每个grad_level的比例分配 [level0_ratio, level1_ratio, level2_ratio]
        """ 
        grads_norm = self.offset_gradient_accum_lod / (self.offset_denom_lod+ 1e-6)
        offset_mask = (self.offset_denom_lod >= 1).squeeze(dim=1)
        grads_norm[~offset_mask] = 0.0
        
        #anchor_grads = torch.sum(grads_norm.reshape(-1, self.n_offsets), dim=-1) / ( torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6 )
        # 计算均值
        anchor_grads = torch.sum(grads_norm.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        # 计算最大值
        grad_dist = anchor_grads* (float(self.fork) ** ((self.get_level_before))).float().squeeze()#*100.0 #*1000.0 #*100.0  #
        # 初始化
        self._grad_level = torch.zeros_like(grad_dist, dtype=torch.int)

        # 初始化
        self._grad_level = torch.zeros_like(grad_dist, dtype=torch.int)
        self._grad_level[grad_dist > self.grad_threhold] = 1

 
 


    def anchor_grad_2frame(self, xyz_save):
        """
        Args:
            grads: 梯度
            num_levels: grad_level的数量（固定为3）
            level_ratios: 每个grad_level的比例分配 [level0_ratio, level1_ratio, level2_ratio]
        """ 
        grads_norm = self.offset_gradient_accum_lod / (self.offset_denom_lod+ 1e-6)
        offset_mask = (self.offset_denom_lod >= 1).squeeze(dim=1)
        grads_norm[~offset_mask] = 0.0
        
        #anchor_grads = torch.sum(grads_norm.reshape(-1, self.n_offsets), dim=-1) / ( torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6 )
        # 计算均值
        anchor_grads = torch.sum(grads_norm.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        # 计算最大值
        grad_dist = anchor_grads* (float(self.fork) ** ((self.get_level_before))).float().squeeze()#*100.0 #*1000.0 #*100.0  #
        # 初始化
        self._grad_level = torch.zeros_like(grad_dist, dtype=torch.int)

        # 按level分组处理     
        # 只处理正梯度的点
        level_grads = grad_dist.reshape(-1).detach().cpu().numpy()
        all_values = grad_dist * (self.get_level_before == torch.floor(torch.tensor(self.levels / 2.0))).squeeze() 

        pos_mask = all_values > 0
        pos_mask = pos_mask.cpu().numpy() 
        pos_grads = level_grads[pos_mask].reshape(-1, 1)  # [M, 1]       
        pt_grad = PowerTransformer(method='yeo-johnson', standardize=True)
        norm_grads = pt_grad.fit_transform(pos_grads)
        alpha = 1.0
        norm_grads = np.sign(norm_grads) * (np.abs(norm_grads) ** alpha)

        # 可以调整权重来平衡坐标和梯度的重要性
        grad_weight = 1.0  # 梯度权重（可调）
        features = np.concatenate([norm_grads * grad_weight], axis=1)
        
        # GMM聚类
        gmm = GaussianMixture(
            n_components=2, 
            covariance_type='full',
            init_params='kmeans', 
            n_init=3, 
            reg_covar=1e-4
        ).fit(features)
        
        # 梯度在最后一维（索引3）
        high_grad_cluster = np.argmax(gmm.means_[:, -1])  
        labels = gmm.predict_proba(features)[:, high_grad_cluster]

        # 假设 labels 是已经通过 GMM 计算出的概率分布
        threshold = np.percentile(labels, self.gmm_threhold)  # 获取前15%的阈值，85%的位置代表前15%
        # 筛选出大于该阈值的点
        pred = np.zeros_like(level_grads, dtype=bool)
        pred[pos_mask] = (labels > threshold)

        # 获取大于阈值的点的索引
        big_score = np.nonzero(pred)[0]
        # 赋值
        self._grad_level[torch.from_numpy(big_score).to(self._grad_level.device)] = 1
        self.expand_grad_by_neighbors_all(xyz_save)
 
    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask,frame_idx=1):
        init_length = self.get_anchor.shape[0] # 记录初始锚点数
        grads[~offset_mask] = 0.0 # 清零不可靠的梯度
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6) #计算每个锚点的平均梯度

        for cur_level in range(self.levels):
            update_value = self.fork ** update_ratio # 计算层级更新因子
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork
            # update threshold
            cur_threshold = threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)

            if len(self.coarse_intervals) > 0:
                if iteration > self.coarse_intervals[-1]:
                    self._extra_level += extra_up * candidate_extra_mask.float()
            #else:
                #print("Warning: coarse_intervals is empty, skipping extra_level update")
                
            #print(self.get_offset.size()) #orch.Size([22160, 3, 10]) 
            #print(self.get_offset.size()) #torch.Size([22160, 10, 3])
            #print(self.get_scaling.size()) #torch.Size([221600])
            t_offset = self.get_offset.transpose(1, 2)#.view([-1, 3])
            all_xyz = self.get_anchor.unsqueeze(dim=1) + t_offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            #print(all_xyz.size()) #torch.Size([22160, 10, 3])
            #print(candidate_mask.size()) #torch.Size([221600])

            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/cur_size).int()
            selected_xyz = all_xyz.reshape([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                #去重检测
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size+self.init_pos
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            if len(self.coarse_intervals) > 0 and (iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size).int()
                selected_xyz_ds = all_xyz.reshape([-1, 3])[candidate_ds_mask]
                selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/ds_size).int()
                selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
                if selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')

            # 创建新锚点的所有属性，特征从附近锚点继承
            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
                
                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda() #,self.latent_decoders["f_rest"].latent_dim
                #current_feat_dim = self._anchor_feat.shape[-1]  # 获取实际的特征维度
        
                # 使用实际的特征维度进行 view 操作
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat_ds = torch.zeros([candidate_anchor_ds.shape[0], self.feat_dim], dtype=torch.float, device='cuda')
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)
                
                # 如果需要转换到 latent 空间
                if isinstance(self.latent_decoders["f_feature_anchor"], LatentDecoderRes):
                    # 如果当前维度和 latent_dim 不同，需要进行转换
                    if self.feat_dim != self.latent_decoders["f_feature_anchor"].latent_dim:
                        # 可以选择：1) 使用零初始化，或 2) 使用 invert 函数（如果可用）
                        latent_dim = self.latent_decoders["f_feature_anchor"].latent_dim
                        new_feat = torch.zeros((new_feat.shape[0], latent_dim), dtype=torch.float, device='cuda')

                '''if isinstance(self.latent_decoders["f_feature_anchor"], LatentDecoderRes):
                    feat_dim = self.latent_decoders["f_feature_anchor"].latent_dim
                else:
                    feat_dim = self.feat_dim
                
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat_ds = torch.zeros([candidate_anchor_ds.shape[0], feat_dim], dtype=torch.float, device='cuda')
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)'''

                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1,2]).float().cuda()*ds_size # *0.05
                new_scaling1 = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling1)
                
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.05 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities_ds = inverse_sigmoid(0.05 * torch.ones((candidate_anchor_ds.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities = torch.cat([new_opacities, new_opacities_ds], dim=0)

                # 修正 offset 的创建方式
                # 方法1: 直接创建正确维度
                new_offsets0 = torch.zeros((candidate_anchor.shape[0], 3, self.n_offsets)).float().cuda()
                new_offsets_ds = torch.zeros((candidate_anchor_ds.shape[0], 3, self.n_offsets)).float().cuda()
                new_offsets_combined = torch.cat([new_offsets0, new_offsets_ds], dim=0)
                
                # 转换为 latent 空间格式
                if isinstance(self.latent_decoders["offset"], LatentDecoderRes):
                    # 如果使用 LatentDecoder，需要正确的维度
                    #new_offsets_flat = new_offsets_combined.transpose(1, 2).flatten(start_dim=1)  # [N, n_offsets*3]
                    new_offsets = torch.zeros((new_offsets_combined.shape[0], 
                                                    self.latent_decoders["offset"].latent_dim), 
                                                    dtype=torch.float, device='cuda')
                else:
                    # 如果是 Identity decoder，保持原始格式
                    new_offsets = new_offsets_combined.contiguous()  # [N, n_offsets, 3] .transpose(1, 2)
                '''new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=2).repeat([1,1,self.n_offsets]).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=2).repeat([1,1,self.n_offsets]).float().cuda()
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)'''

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device='cuda')
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])
                new_extra_level_grad = torch.zeros(new_level.shape[0], dtype=torch.bool, device="cuda")

                new_flow = torch.zeros_like(new_anchor)

                #将新锚点添加到优化器
                d1 = {
                    "rotation": new_rotation,
                    "opacity": new_opacities,
                }   
                d2 = {
                    "anchor": new_anchor,
                    "sc": new_scaling,
                    "f_feature_anchor": new_feat,
                    "flow": new_flow,
                    "offset": new_offsets,
                }   

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                self.anchor_gradient_accum = torch.cat([self.anchor_gradient_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0) #self.anchor_gradient_accum[valid_points_mask]
                self.denom = torch.cat([self.denom, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0) #self.denom[valid_points_mask]
                self.denom_lod = torch.cat([self.denom_lod, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0) #self.denom_lod[valid_points_mask] 
 
                torch.cuda.empty_cache()
                optimizable_tensors, optimizable_tensors2 = self.cat_tensors_to_optimizer(d1, d2)
                self.lod_rotation = optimizable_tensors["rotation"]
                self.lod_opacity = optimizable_tensors["opacity"]

                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)
                for param in self.param_names:
                    self._latents[param] = optimizable_tensors2[param]

                if frame_idx==1:
                    self._anchor_mask_grad = torch.cat([self._anchor_mask_grad, new_extra_level_grad], dim=0)
                    self._anchor_mask_grad_report = torch.cat([self._anchor_mask_grad_report, new_extra_level_grad], dim=0)

                        # ===== 新增：同步更新 LatentDecoderRes 的 decoded_att =====
                for param_name in ["anchor", "f_feature_anchor", "sc", "flow", "offset"]:
                    decoder = self.latent_decoders[param_name]
                    if isinstance(decoder, LatentDecoderRes) and hasattr(decoder, 'decoded_att'):
                        # 为新增的锚点创建对应的 decoded_att
                        if param_name == "anchor":
                            new_decoded = new_anchor  # 使用新锚点的坐标
                        elif param_name == "f_feature_anchor":
                            new_decoded = torch.zeros((new_feat.shape[0], self.feat_dim), dtype=torch.float, device='cuda')
                        elif param_name == "sc":
                            new_decoded = new_scaling1  # 使用新的 new_scaling0
                        elif param_name == "flow":
                            new_decoded = torch.zeros_like(new_anchor)
                        elif param_name == "offset":
                            new_decoded = torch.zeros((new_anchor.shape[0], 30), dtype=torch.float, device='cuda') 
                        
                        # 拼接到现有的 decoded_att
                        decoder.decoded_att = torch.cat([decoder.decoded_att, new_decoded], dim=0)

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.eval()
            unified_mlp = torch.jit.trace(self.unified_mlp, (torch.rand(1, self.feat_dim+self.view_dim).cuda()))
            unified_mlp.save(os.path.join(path, 'unified_mlp.pt'))
            if self.appearance_dim > 0:
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
            self.train()
        elif mode == 'unite':
            param_dict = {}
            param_dict['unified_mlp'] = self.unified_mlp.state_dict()
            torch.save(param_dict, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.unified_mlp = torch.jit.load(os.path.join(path, 'unified_mlp.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.unified_mlp.load_state_dict(checkpoint['unified_mlp'])
        else:
            raise NotImplementedError

 
    def copy(self):
        """Create a deep copy of the GaussianModel instance.
        
        Returns:
            GaussianModel: A new instance with copied attributes and parameters.
        """
        # Create new instance with same initialization parameters
        new_model = GaussianModel(self.max_sh_degree, self.latent_args, self.model_args, self.frame_idx, self.use_xyz_legacy)
        
        # Copy basic attributes
        new_model.active_sh_degree = self.active_sh_degree
        new_model.spatial_lr_scale = self.spatial_lr_scale
        new_model.percent_dense = self.percent_dense
        
        # Copy latents
        for param_name in self.param_names:
            new_model._latents[param_name] = nn.Parameter(self._latents[param_name].data.clone().requires_grad_(True))
        
        # Copy decoder state dicts
        atts = self.get_atts # All the latent variables
        for i, att_name in enumerate(atts):
            if not isinstance(self.latent_decoders[param_name], DecoderIdentity):
                # first check if the decoder is a DecoderIdentity
                if isinstance(new_model.latent_decoders[param_name], DecoderIdentity):
                    decoder = LatentDecoderRes(
                        latent_dim=self.latent_args.latent_dim[i],
                        feature_dim=self.feature_dims[att_name],
                        ldecode_matrix=self.latent_args.ldecode_matrix[i],
                        latent_norm=self.latent_args.latent_norm[i],
                        num_layers_dec=self.latent_args.num_layers_dec[i],
                        hidden_dim_dec=self.latent_args.hidden_dim_dec[i],
                        activation=self.latent_args.activation[i],
                        use_shift=self.latent_args.use_shift[i],
                        ldec_std=self.latent_args.ldec_std[i],
                        final_activation=self.latent_args.final_activation[i],
                    ).cuda()
                    new_model.latent_decoders[att_name] = decoder
                else: 
                    new_model.latent_decoders[param_name].load_state_dict(
                        self.latent_decoders[param_name].state_dict()
                    )
        
        # Copy masks
        new_model.mask_xyz.data = self.mask_xyz.data.clone()
        new_model.mask_anchor.data = self.mask_anchor.data.clone()
        new_model.mask_features_dc.data = self.mask_features_dc.data.clone()
        new_model.mask_features_rest.data = self.mask_features_rest.data.clone()
        new_model.mask_scaling.data = self.mask_scaling.data.clone()
        new_model.mask_rotation.data = self.mask_rotation.data.clone()
        new_model.mask_opacity.data = self.mask_opacity.data.clone()
        
        # Copy freeze states
        new_model.frz_xyz = self.frz_xyz
        new_model.frz_anchor = self.frz_anchor
        new_model.frz_features_dc = self.frz_features_dc
        new_model.frz_features_rest = self.frz_features_rest
        new_model.frz_scaling = self.frz_scaling
        new_model.frz_rotation = self.frz_rotation
        new_model.frz_opacity = self.frz_opacity
        
        # Copy previous attributes and latents
        for param_name in self.param_names:
            if self.prev_atts[param_name] is not None:
                new_model.prev_atts[param_name] = self.prev_atts[param_name].clone()
            if self.prev_latents[param_name] is not None:
                new_model.prev_latents[param_name] = self.prev_latents[param_name].clone()
        
        # Copy gate attributes if they exist
        if self.gate_atts is not None:
            new_model.gate_atts = self.gate_atts.copy()
            new_model.gate_params = self.gate_params.copy()
        
        # Copy other tensors
        new_model.max_radii2D_lod = self.max_radii2D_lod.clone()
        new_model.xyz_gradient_accum = self.xyz_gradient_accum.clone()
        new_model.anchor_gradient_accum = self.anchor_gradient_accum.clone()
        new_model.infl_accum = self.infl_accum.clone()
        new_model.denom = self.denom.clone()
        new_model.denom = self.denom_lod.clone()
        new_model.infl_denom = self.infl_denom.clone()
        
        # Copy mapping and xyz_before if they exist and are not None
        if hasattr(self, 'mapping') and self.mapping is not None:
            new_model.mapping = self.mapping.clone()
        if hasattr(self, 'anchor_before') and self.anchor_before is not None:
            new_model.anchor_before = self.anchor_before.clone()
        
        # Copy added_mask if it exists and is not None
        if hasattr(self, 'added_mask') and self.added_mask is not None:
            new_model.added_mask = self.added_mask.clone()
        
        # Copy init_probs if it exists and is not None
        if hasattr(self, 'init_probs') and self.init_probs is not None:
            new_model.init_probs = self.init_probs.clone()
        
        return new_model


    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        self.voxel_size = torch.tensor(plydata.elements[0]["info"][0]).float()
        self.standard_dist = torch.tensor(plydata.elements[0]["info"][1]).float()

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
    
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self.lod_scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self.lod_opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self.lod_rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.levels = torch.max(self._level) - torch.min(self._level) + 1


    def load_mlp_checkpoints0(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.unified_mlp_static = torch.jit.load(os.path.join(path, 'unified_mlp.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.unified_mlp_static.load_state_dict(checkpoint['unified_mlp'])
        else:
            raise NotImplementedError

    def load_ply_sparse_gaussian0(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(np.int)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        self.voxel_size = torch.tensor(plydata.elements[0]["info"][0]).float()
        self.standard_dist = torch.tensor(plydata.elements[0]["info"][1]).float()

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        self.get_anchor_feat_before = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self.get_level_before = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._offset_before = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self.get_anchor_before = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self.get_scaling_before = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
