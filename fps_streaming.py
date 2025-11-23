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
from scene import Scene
import os
import yaml
import socket
import sys
import hashlib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import torch.utils.benchmark as benchmark
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from utils.loader_utils import MultiViewVideoDataset, SequentialMultiviewSampler
from scene.decoders import DecoderIdentity
from arguments import ModelParams, PipelineParams, OptimizationParams, QuantizeParams, OptimizationParamsInitial, OptimizationParamsRest, get_combined_args
import json

try:
    import wandb
    if not ('SLURM_PROCID' in os.environ and os.environ['SLURM_PROCID']!='0'):
        WANDB_FOUND = True
    else:
        WANDB_FOUND = False
except ImportError:
    WANDB_FOUND = False


def render_set(views, gaussians, pipeline, background):

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]

def render_fn(views, gaussians, pipeline, background, use_amp):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        for view in views:
            render(view, gaussians, pipeline, background)

def measure_fps(scene, gaussians, pipeline, background, use_amp=False):
    with torch.no_grad():
        views = scene.getTrainCameras() + scene.getTestCameras()
        t0 = benchmark.Timer(stmt='render_fn(views, gaussians, pipeline, background, use_amp)',
                            setup='from __main__ import render_fn',
                            globals={'views': views, 'gaussians': gaussians, 'pipeline': pipeline, 
                                    'background': background, 'use_amp': use_amp},
                            )
        time = t0.timeit(100)
        fps = len(views)/time.median
        print("Rendering FPS: ", fps)
    return fps
        

def measure_fps_decode(gaussians):
    with torch.no_grad():
        t0 = benchmark.Timer(stmt='gaussians.get_decoded_atts',
                            globals={'gaussians': gaussians},
                            )
        time = t0.timeit(100)
        fps = 1/time.median
        print("Decoding FPS: ", fps)
    return fps

def generate_test_indices(total_cameras, llff_number):
    """
    根据总相机数量和llff_number生成测试索引
    
    Args:
        total_cameras (int): 总相机数量
        llff_number (int): llff间隔数
    
    Returns:
        torch.Tensor: 测试索引张量
    """
    # 生成所有索引 (0到total_cameras-1)
    all_indices = torch.arange(total_cameras)
    
    # 生成要排除的索引（每隔llff_number个索引）
    exclude_indices = torch.arange(0, total_cameras, llff_number)
    
    # 创建布尔掩码，标记哪些索引要保留
    mask = torch.ones(total_cameras, dtype=torch.bool)
    mask[exclude_indices] = False
    
    # 使用掩码获取测试索引
    test_indices = all_indices[mask]
    
    return test_indices

def render_sets(dataset: ModelParams, opt: OptimizationParams, pipeline: PipelineParams, qp:QuantizeParams, args,
             skip_train: bool, skip_test: bool):
    
    with torch.no_grad():
        test_indices = generate_test_indices(args.total_cameras, dataset.llffnumber)

        if not skip_train:
            # Create dataset and loader for training and testing at each time instance
            train_image_dataset = MultiViewVideoDataset(dataset.source_path, split='train', test_indices=test_indices,
                                                        max_frames=dataset.max_frames, start_idx=0)
            train_sampler = SequentialMultiviewSampler(train_image_dataset)
            train_loader = iter(torch.utils.data.DataLoader(train_image_dataset, batch_size=train_image_dataset.n_cams, 
                                                            sampler=train_sampler, num_workers=4))
        
        if not skip_test:
            test_image_dataset = MultiViewVideoDataset(dataset.source_path, split='test', test_indices=test_indices, 
                                                    max_frames=dataset.max_frames, start_idx=0)
            test_sampler = SequentialMultiviewSampler(test_image_dataset)
            test_loader = iter(torch.utils.data.DataLoader(test_image_dataset, batch_size=test_image_dataset.n_cams, 
                                                            sampler=test_sampler, num_workers=4))
        
        start_frame_idx = dataset.start_idx +1
        if not skip_train:
            train_data = next(train_loader)
            train_images, train_paths = train_data
        if not skip_test:
            try:
                test_data = next(test_loader)
                test_images, test_paths = test_data
            except StopIteration:
                print('No test cameras found, disabling testing.')
                test_images, test_paths = None, None

        if not skip_train:
            train_image_data = {'image':train_images.cuda(),'path':train_paths,'frame_idx':0}
        else:
            train_image_data = None
        if not skip_test:
            test_image_data = {'image':test_images.cuda(),'path':test_paths,'frame_idx':0}
        else:
            test_image_data = None

        # Create the gaussian model and scene, initialized with frame 1 images from dataset
        qp.seed = dataset.seed

        gaussians = GaussianModel(sh_degree=dataset.sh_degree, latent_args=qp, model_args=dataset, feat_dim=dataset.feat_dim,
        n_offsets=dataset.n_offsets, fork=dataset.fork, grad_threhold=args.grad_threhold, use_feat_bank=dataset.use_feat_bank, appearance_dim=dataset.appearance_dim, 
        add_opacity_dist=dataset.add_opacity_dist, add_cov_dist=dataset.add_cov_dist, add_color_dist=dataset.add_color_dist, add_level=dataset.add_level, 
        visible_threshold=dataset.visible_threshold, dist2level=dataset.dist2level, grad_threthold_mode=args.grad_threthold_mode, base_layer=dataset.base_layer, progressive=dataset.progressive, extend = dataset.extend)
  
        max_frames = args.max_frames
        # Setup training arguments
        gaussians.eval()
        gaussians.frame_idx = start_frame_idx
        fmodel_path0 = os.path.join(dataset.model_path, 'frames','0001')
        model_path = os.path.join(dataset.model_path,'frames',str(start_frame_idx).zfill(4))
        # Setup training arguments

        scene = Scene(
            dataset,
            gaussians,
            fmodel_path0,
            model_path,
            argstest = args,
            load_iteration=-1, 
            train_image_data= train_image_data,
            test_image_data=test_image_data,
            shuffle=False, 
            verbose=False, 
            N_video_views=max_frames
        )

        opt.set_params(start_frame_idx)
        gaussians.training_setup(opt)


        #checkpoint_path = os.path.join(args.model_path,'frames',str(start_frame_idx).zfill(4), 'ckpt.pth')
        #print('Loading checkpoint at ', checkpoint_path)
        #model_params, iteration, start_frame_idx, training_metrics = torch.load(checkpoint_path)

        #gaussians.restore_fps(model_params, opt, start_frame_idx)
        #gaussians.frame_idx = start_frame_idx

        '''for param_name in gaussians.param_names:
            if gaussians.gate_params[param_name]:
                att = gaussians.get_decoded_atts[param_name]
                latents = gaussians.get_atts[param_name]
                latents.data = att.data
                gaussians.latent_decoders[param_name] = DecoderIdentity()
                gaussians.gate_params[param_name] = False
        gaussians.gate_atts = None'''
                
        #scene.model_path = os.path.join(args.model_path,'frames',str(start_frame_idx).zfill(4))
        scene.updateCameraImages(args, train_image_data, test_image_data, start_frame_idx, resolution_scales=[1.0])

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        fps = measure_fps(scene, gaussians, pipeline, background, use_amp=False)
        return fps, None
 
if __name__ == "__main__":

    print('Running on ', socket.gethostname())
    # Config file is used for argument defaults. Command line arguments override config file.
    # testing
    config_path = sys.argv[sys.argv.index("--config")+1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser, config['model_params'])
    op_i = OptimizationParamsInitial(parser, config['opt_params_initial'])
    op_r = OptimizationParamsRest(parser, config['opt_params_rest'])
    pp = PipelineParams(parser, config['pipe_params'])
    qp = QuantizeParams(parser, config['quantize_params'])

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument('--total_cameras', type=int, default=13, help='Total number of cameras')
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--grad_threhold", type=float, default=0.0000)
    parser.add_argument('--grad_threthold_mode', type=str, default="Gmm") #Gmm or hard_threshold
    args = parser.parse_args(sys.argv[1:])
    
    # Merge optimization args for initial and rest and change accordingly
    op = OptimizationParams(op_i.extract(args), op_r.extract(args))

    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    lp_args = lp.extract(args)
    pp_args = pp.extract(args)
    qp_args = qp.extract(args)

    fps_all = []
    for start_idx in range(1, lp_args.max_frames-1):
        print("FPS for frame ", start_idx)
        lp_args.start_idx = start_idx
        fps, decode_fps = render_sets(lp_args, op, pp_args, qp_args, args, args.skip_train, args.skip_test)
        fps_all.append(fps)
        if start_idx == lp_args.max_frames//2:
            fps_mid_frame = fps

    wandb_enabled = WANDB_FOUND and lp_args.use_wandb
    
    print("Mean FPS: ", np.array(fps_all).mean())
    print("Median FPS: ", np.median(fps_all))
    print("FPS median frame: ", fps_mid_frame)
    
    # 从模型路径中提取文件名
    model_name = os.path.basename(args.model_path.rstrip('/'))  # 获取 discussion3_r0.0_llff3_eprest6

    # 创建结果字典
    fps_results = {
        "model_path": args.model_path,
        "model_name": model_name,
        "mean_fps": float(np.array(fps_all).mean()),
        "median_fps": float(np.median(fps_all)),
        "fps_median_frame": float(fps_mid_frame),
        "all_fps": [float(fps) for fps in fps_all],  # 保存所有帧的FPS以便后续分析
        "total_frames": len(fps_all)
    }


    # 设置输出目录和文件名
    output_dir = "./fps_results"  # 你可以修改为想要的输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_fps.json")

    # 保存到JSON文件
    with open(output_file, 'w') as f:
        json.dump(fps_results, f, indent=4)

    print(f"FPS results saved to: {output_file}")