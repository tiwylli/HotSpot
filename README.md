# ***HotSpot***: Screened Poisson Equation for Signed Distance Function Optimization

This repository contains the code for the paper [HotSpot: Screened Poisson Equation for Signed Distance Function Optimization](https://arxiv.org/abs/2411.14628).

Please follow the installation instructions below.

## Installation

Our codebase uses [PyTorch](https://pytorch.org/). The code was tested with Python 3.9.19, torch 2.4.1, tensorboardX 2.6.2.2, CUDA 11.8 on Ubuntu 20.04.6 LTS. 

We also provide a [docker image](https://hub.docker.com/layers/galaxeaaa/pytorch-cuda11.8/latest/images/sha256-5e32b788a2cb0740234a7ed166451f4324cd79e07add2e7d61569013faa3c0e0?context=repo) that pre-installed all the requirements above in the conda environment named `torch`. Use `/workspace/conda init <your_shell>` to initialize the conda environment for your shell. Then follow the instructions below to activate the environment and install the requirements. For a full list of requirements see [the `requirement.txt` file](requirements.txt). Note we also use `plotly-orca` for visualisation, which needs to be installed from conda.

Example installation code if you are using the docker image:
```sh
/workspace/conda init <your_shell> # Change <your_shell> to your shell, e.g. bash, zsh, fish
conda activate torch
conda install pip
pip install -r requirements.txt
conda install -y -c plotly plotly plotly-orca 
```

Example installation code if you are **not** using the docker image:
```sh
conda create -n torch python=3.9
conda activate torch
conda install pip
pip install -r requirements.txt
conda install -y -c plotly plotly plotly-orca
# Install with instructions from https://pytorch.org/get-started/locally/
# Below is instructions for installation of latest version of PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

To compute metrics for ShapeNet, you will need to download

## Usage

### 1. Testing on 2D Shapes (No External Data required)

We inherit from [StEik](https://github.com/sunyx523/StEik) a 2D shape dataset generator (`./dataset/shape_base.py` and `./dataset/shape_2d.py`) that includes three shapes: Circle, L shape polygon, and Koch snowflake. We also designed 10 more shapes with more complex topology to test our idea. The code generally allows any polygonal shape to be used and can be extended to other 2D shapes. 

To train a 2D shape neural representation and reconstruct the surface (curve in this case) for all the shapes run the script 
```sh
bash ./scripts/run_spin_2d.sh
```

After training, use the following script to compute the metrics
```sh
bash ./scripts/run_metric_2d.sh PATH_TO_EXPERIMENT
```

### 2. Surface Reconstruction (and Scene Reconstruction)
#### 2.1 Data for Surface Reconstruction
##### 2.1.1 Surface Reconstruction Benchamark data
The Surface Reconstruction Benchmark (SRB) data is provided in the [Deep Geometric Prior repository](https://github.com/fwilliams/deep-geometric-prior).
This can be downloaded via terminal into the data directory by running `scripts/download_srb.sh` (1.12GB download). We use the entire dataset (of 5 complex shapes).

If you use this data in your research, make sure to cite the Deep Geometric Prior paper.

##### 2.1.2 ShapeNet data
We use a subset of the [ShapeNet](https://shapenet.org/) data as chosen by [Neural Splines](https://github.com/fwilliams/neural-splines). This data is first preprocessed to be watertight as per the pipeline in the [Occupancy Networks repository](https://github.com/autonomousvision/occupancy_networks), who provide both the pipleline and the entire preprocessed dataset (73.4GB). 

The Neural Spline split uses the first 20 shapes from the test set of 13 shape classes from ShapeNet. We provide [a subset of the ShapeNet preprocessed data](https://drive.google.com/file/d/1h6TFHnza0axOZz5AuRkfyLMx_sFcu_Yf/view?usp=sharing) (the subset that corresponds to the split of Neural Splines) and [the resulting point clouds for that subset](https://drive.google.com/file/d/14CW_a0gS3ARJsIonyqPc5eKT3iVcCWZ0/view?usp=sharing).

In addition to the above subset and resulting point clouds provided by DiGS and StEik, we ran the data preprocessing pipeline introduced in [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks) and provide [the watertight ShapeNet meshes for this subset](https://drive.google.com/file/d/1HAZ41-rZQIw_pezj-ES-ZtgXO6JanU-V/view?usp=sharing) (376.1MB). This is for our distance metrics computation. Note that the watertight meshes are all translated and scaled to exactly match the point clouds provided by DiGS and StEik.

These can be downloaded via terminal into the data directory by running `scripts/download_shapenet.sh` (783.76MB download).

If you use this data in your research, make sure to cite the ShapeNet and Occupancy Network papers, and if you report on this split, compare and cite to the Neural Spline paper.

##### 2.1.3 Scene Reconstruction data
For scene reconstruction, we used the [scene from the SIREN paper](https://drive.google.com/drive/folders/1_iq__37-hw7FJOEUK1tX7mdp8SKB368K?usp=sharing). This can be downloaded via terminal into the data directory by running `scripts/download_scene.sh`  (56.2MBMB download).

If you use this data in your research, make sure to cite the SIREN paper.

#### 2.2 Running Surface Reconstruction
To train, test and evaluate on ShapeNet run 

```bash scripts/run_hotspot_shapenet.sh```

Similarly we provide a script for SRB: 

```bash scripts/run_hotspot_srb.sh```

and for scene reconstruction:

```bash scripts/run_hotspot_scene.sh``` 

These scripts take a config file located in `./configs/` as an argument. The config files are named `hotspot_shapenet.yaml`, `hotspot_srb.yaml`, and `hotspot_scene.yaml` respectively.

## Acknowledgements

Our code is built on top of the code from [DiGS](https://github.com/Chumbyte/DiGS) and [StEik](https://github.com/sunyx523/StEik), but is significantly modified.

## License and Citation

If you find our work useful in your research, please cite our paper:

[Preprint](https://arxiv.org/abs/2411.14628):
```bibtex
@misc{wang2024hotspotscreenedpoissonequation,
      title={HotSpot: Screened Poisson Equation for Signed Distance Function Optimization}, 
      author={Zimo Wang and Cheng Wang and Taiki Yoshino and Sirui Tao and Ziyang Fu and Tzu-Mao Li},
      year={2024},
      eprint={2411.14628},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.14628}, 
}
```

See [LICENSE](LICENSE) file.