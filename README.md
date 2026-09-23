# LoD-Structured 3D Gaussian Splatting for Streaming Video Reconstruction

**StreamLoD-GS** StreamLoD-GS, an LoD-based Gaussian Splatting framework designed specifically for SFVV. Our approach includes: 1) an Anchor- and Octree-based LoD-structured 3DGS with a hierarchical Gaussian dropout technique to ensure efficient and stable optimization while maintaining high-quality rendering; 2) a GMM-based motion partitioning mechanism that separates dynamic and static content, refining dynamic regions while preserving background stability; and 3) a quantized residual refinement framework that reduces storage requirements without compromising visual quality. Extensive experiments demonstrate that StreamLoD-GS achieves competitive or state-of-the-art performance in terms of quality, efficiency, and storage, outperforming related methods.

 
 
## Code Setup

### 0. Dependencies

We have only tested on Linux environments with CUDA 11.6+ compatible systems. 

### Clone and Setup Repo

Our software has some submodules, please clone the repo recursively. 

```bash
git clone --recurse-submodules git@github.com:Xinhui-99/StreamLoD_GS.git StreamLoD_GS
cd StreamLoD_GS
# set up some relevant directories
mkdir data
mkdir logs
mkdir output
```

#### Conda Environment

```bash
# create the conda environment
conda env create -f environment.yml
conda activate StreamLoD-GS

# install submodules
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization

 

## Data Preparation

We assume datasets are organized as follows:

### DyNeRF or Meet Room
```
| --- data
|   | [dataset_directory]
│     | [scene_name] 
│   	  | cam01
|            | images
|     		  | ---0000.png
│     		  | --- 0001.png
│     		  | --- ...
│   	  | cam02
|            | images
│     		  | --- 0000.png
│     		  | --- 0001.png
│     		  | --- ...
│   	  | ...
│   	  | sparse_
│     		  | --- cameras.bin
│     		  | --- images.bin
│     		  | --- ...
│   	  | points3D_downsample2.ply
│   	  | poses_bounds.npy
```

 
If you have a dataset of multi-view videos, please follow the steps to produce the required data:

1. organize the videos into folders of images for each camera view
2. follow the instructions in the [VGGT](https://github.com/facebookresearch/vggt) codebase to create the COLMAP camera data at the `sparse_` directory

For more information on how datasets are loaded, please see `scene/dataset_readers.py`.

---

## 💻 Training

You can train a scene by running:

```bash
python train.py --config [config_path] -s [source_path] -m [output_name]
```

For example:
```bash
python train.py --config configs/dynerf.yaml -s data/dynerf/coffee_martini -m ./output/coffee_martini_trained
```

The training script builds on the original 3DGS training script, and, as such, shares many of the same command line arguments. We add new arguments to control compression hyperparameters and 3DGS training hyperparameters for the initial frame and residual frames. 

Please see specific configuration files in `configs` for examples, and `arguments/__init__.py` for the full list of arguments.

<details>
<summary><span style="font-weight: bold;">Useful Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```0``` by default.
  #### --max_frames
  Maximum number of frames to process, ```300``` by default.
  #### --log_images
  Flag to save rendered images during training.
  #### --log_ply
  Flag to save point cloud in PLY format during training.
  #### --log_compressed
  Flag to save compressed model during training.
  #### --save_format
  Format to save the model in, ```ply``` by default.
 
</details>
<br>


---

## 🎥 Rendering and Evaluation

### Rendering from Dense 3DGS

```bash
python render.py -s <path to scene> -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

We also provide a script to render spiral viewpoints.
```bash
python render_fvv.py  -s <path to scene> -m <path to trained model> # Generate renderings
```

Example usage:
```
# Train coffee-martini scene
python train.py --config configs/dynerf.yaml --log_images --log_ply -s data/dynerf/coffee_martini -m ./output/coffee_martini_trained 

# Compute metrics
python metrics_video.py -m ./output/coffee_martini_trained 

# Render static camera viewpoints and spiral
python render.py -s data/dynerf/coffee_martini -m ./output/coffee_martini_trained
python render_fvv.py --config configs/dynerf.yaml  -s data/dynerf/coffee_martini -m ./output/coffee_martini_trained
```

 
 **Citation**

If you use this code, please cite:
```python
@article{liu2026lod,
  title={LoD-Structured 3D Gaussian Splatting for Streaming Video Reconstruction},
  author={Liu, Xinhui and Wang, Can and Liu, Lei and Chen, Zhenghao and Jiang, Wei and Wang, Wei and Xu, Dong},
  journal={arXiv preprint arXiv:2601.18475},
  year={2026}
}
```

