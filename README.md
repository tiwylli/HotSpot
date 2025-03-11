# ***HotSpot***: Signed Distance Function Optimization with an Asymptotically Sufficient Condition

[Zimo Wang](https://zeamoxwang.github.io/homepage/)\*, [Cheng Wang](https://galaxeaaa.github.io/)\*, [Taiki Yoshino](https://www.linkedin.com/in/taiki-yoshino-167a60266), [Sirui Tao](https://dylantao.github.io/), [Ziyang Fu](https://fzy28.github.io/), [Tzu-Mao Li](https://cseweb.ucsd.edu/~tzli/) (* denotes equal contribution)

[Paper](https://arxiv.org/abs/2411.14628) | [Project Page](https://zeamoxwang.github.io/HotSpot-CVPR25/) | [Code](https://github.com/Galaxeaaa/HotSpot)

Please follow the installation instructions below.

## Installation

Our code was tested with Python 3.9.19, torch 2.4.1, tensorboardX 2.6.2.2, CUDA 11.8 on Ubuntu 20.04.6 LTS. 

We also provide a [docker image](https://hub.docker.com/layers/galaxeaaa/pytorch-cuda11.8/latest/images/sha256-5e32b788a2cb0740234a7ed166451f4324cd79e07add2e7d61569013faa3c0e0?context=repo) that pre-installed all the requirements above in the conda environment named `torch`. Use `/workspace/conda init <your_shell>` to initialize the conda environment for your shell. Then follow the instructions below to activate the environment and install the requirements. For a full list of requirements see [the `requirement.txt` file](requirements.txt). Note we also use `plotly` for visualisation, which needs to be installed from conda.

Example installation code if you **are** using the docker image:
```sh
/workspace/conda init <your_shell> # Change <your_shell> to your shell, e.g. bash, zsh, fish
conda activate torch
pip install -r requirements.txt
conda install -y -c plotly plotly plotly-orca 
```

Example installation code if you **are not** using the docker image:
```sh
conda create -n torch python=3.9
conda activate torch
pip install -r requirements.txt
conda install -y -c plotly plotly plotly-orca
# Install with instructions from https://pytorch.org/get-started/locally/
# Below is instructions for installation of latest version of PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Testing on 2D Shapes (No External Data required)

We inherit from [StEik](https://github.com/sunyx523/StEik) a 2D shape dataset generator (`./dataset/shape_base.py` and `./dataset/shape_2d.py`) that includes three shapes: Circle, L shape polygon, and Koch snowflake. We also designed 10 more shapes with more complex topology to test our idea. The code generally allows any polygonal shape to be used and can be extended to other 2D shapes. 

To train a 2D shape neural representation and reconstruct the surface (curve in this case) for all the shapes run the script 

```sh
bash ./scripts/run_hotspot_2d.sh
```

The script takes a config file located in `./configs/hotspot_2d.toml` as an argument. This will create a folder `./log/2D/HotSpot/` where the results are stored.


After training, use the following script to compute the metrics

```sh
bash ./scripts/run_metric_2d.sh <PATH_TO_EXPERIMENT>
```

If you use the script above, the path to the experiment should be `./log/2D/HotSpot/`.

### 2. Surface Reconstruction
#### 2.1 Data Preparation 
##### 2.1.1 Surface Reconstruction Benchamark data
The Surface Reconstruction Benchmark (SRB) data is provided in the [Deep Geometric Prior repository](https://github.com/fwilliams/deep-geometric-prior), and can be downloaded from [this link](https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view) (1.12 GB).

If you use this data in your research, make sure to cite the Deep Geometric Prior paper.

##### 2.1.2 ShapeNet data
We use a subset of the [ShapeNet](https://shapenet.org/) data as chosen by [Neural Splines](https://github.com/fwilliams/neural-splines). This subset is first preprocessed to be watertight as per the pipeline in the [Occupancy Networks repository](https://github.com/autonomousvision/occupancy_networks), who provide both the pipleline and the entire preprocessed dataset (73.4 GB).

The data you should download for training includes [the preprocessed data](https://drive.google.com/file/d/1h6TFHnza0axOZz5AuRkfyLMx_sFcu_Yf/view?usp=sharing) (412.8 MB) and [the resulting point clouds](https://drive.google.com/file/d/14CW_a0gS3ARJsIonyqPc5eKT3iVcCWZ0/view?usp=sharing) (371 MB) of this subset.

For our distance metrics computation, we ran the data preprocessing pipeline introduced in Occupancy Networks and provide [the watertight meshes](https://drive.google.com/file/d/1HAZ41-rZQIw_pezj-ES-ZtgXO6JanU-V/view?usp=sharing) (376.1 MB) for this subset. The watertight meshes are all translated and scaled to exactly match the point clouds provided above.

If you use this data in your research, make sure to cite the ShapeNet and Occupancy Network papers, and if you report on this split, compare with and cite the Neural Spline paper.

##### 2.1.3 High genus data
We also run experiment on some high genus shapes. It is composed of two parts: [NIE dataset](https://drive.google.com/file/d/12Z0nmNISGlsfy-QK3_uw91uuj9hC6DDc/view?usp=drive_link) (5 shapes, 34.7 MB) used in [A Level Set Theory for Neural Implicit Evolution under Explicit Flows](https://ishit.github.io/nie/index.html), and [nested voronoi spheres](https://drive.google.com/file/d/1LGN6HUrZFKWoRvR2gKmmGzMjFLJbKJP2/view?usp=drive_link) (2 shapes, 27.6 MB).

If you use this data in your research, cite [A Level Set Theory for Neural Implicit Evolution under Explicit Flows](https://ishit.github.io/nie/index.html).

##### 2.1.4 Scene Reconstruction data
For scene reconstruction, we used the [scene from the SIREN paper](https://drive.google.com/drive/folders/1_iq__37-hw7FJOEUK1tX7mdp8SKB368K?usp=sharing).

If you use this data in your research, make sure to cite the SIREN paper.

#### 2.2 Running Surface Reconstruction
To train, test and evaluate on ShapeNet run 

```bash scripts/run_hotspot_shapenet.sh```

Similarly we provide a script for SRB: 

```bash scripts/run_hotspot_srb.sh```

and for high genus shapes:

```bash scripts/run_hotspot_nie.sh```

and for scene reconstruction:

```bash scripts/run_hotspot_scene.sh``` 

These scripts take a config file located in `./configs/` as an argument. The config files are named `hotspot_shapenet.toml`, `hotspot_srb.toml`, and `hotspot_scene.toml` respectively.

### 3. Ray Tracing

Instructions to be added.

## Acknowledgements

Our code is built on top of the code from [DiGS](https://github.com/Chumbyte/DiGS) and [StEik](https://github.com/sunyx523/StEik), but is significantly modified.

## License and Citation

If you find our work useful in your research, please cite our paper:

[Preprint](https://arxiv.org/abs/2411.14628)
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