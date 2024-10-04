# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import configargparse
import torch
import os
import numpy as np


def get_train_args():
    parser = configargparse.ArgParser(description="Local implicit functions experiment.")
    parser.add_argument("--config", is_config_file=True, help="Config file path.")
    # Dataset
    parser.add_argument("--task", type=str, default="3d", help="3d | 2d.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/deep_geometric_prior_data",
        help="Path to dataset folder.",
    )
    parser.add_argument(
        "--raw_dataset_path",
        type=str,
        default="../data/deep_geometric_prior_data",
        help="Path to dataset folder.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="",
        help="Name of file to reconstruct (within the dataset path).",
    )
    # - 2D basic shape dataset
    parser.add_argument(
        "--shape_type", type=str, default="L", help="Shape dataset to load. (circle | square | L |starAndHexagon | button)."
    )
    # Training
    parser.add_argument("--gpu_idx", type=int, default=0, help="Set < 0 to use CPU.")
    parser.add_argument("--log_dir", type=str, default="./log/debug", help="Log directory.")
    parser.add_argument("--seed", type=int, default=3627473, help="Random seed.")
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=5000,
        help="Number of iterations in the generated train and test set.",
    )
    parser.add_argument("--parallel", type=int, default=False, help="Use data parallel.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument(
        "--grad_clip_norm", type=float, default=10.0, help="Value to clip gradients to."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of samples in a minibatch."
    )
    parser.add_argument(
        "--n_points", type=int, default=30000, help="Number of points in each point cloud."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader."
    )

    # Visualization and logging
    parser.add_argument(
        "--results_path",
        type=str,
        default="./log/surface_reconstruction/DiGS_surf_recon_experiment/result_meshes",
        help="Path to results directory.",
    )
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=100,
        help="Number of iterations between visualizations.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Number of iterations between logging.",
    )
    parser.add_argument(
        "--vis_grid_range",
        type=float,
        default=1.2,
        help="Range of the grid to sample points while visualizing.",
    )
    parser.add_argument(
        "--vis_grid_res", type=int, default=512, help="Grid resolution for reconstruction."
    )
    parser.add_argument(
        "--vis_contour_interval",
        type=float,
        default=0.05,
        help="Interval for level set visualization.",
    )
    parser.add_argument(
        "--vis_normals", type=bool, default=False, help="Indicator to visualize normals."
    )
    parser.add_argument(
        "--n_vis_normals",
        type=int,
        default=100,
        help="Number of normals to visualize.",
    )
    parser.add_argument(
        "--vis_heat", type=bool, default=False, help="Indicator to visualize heat."
    )
    parser.add_argument(
        "--save_video", type=bool, default=False, help="Indicator to save video."
    )
    parser.add_argument(
        "--video_fps", type=int, default=6, help="Frames per second for video."
    )

    # Network architecture and loss
    parser.add_argument(
        "--decoder_hidden_dim", type=int, default=256, help="Length of decoder hidden dim."
    )
    parser.add_argument(
        "--encoder_hidden_dim", type=int, default=128, help="Length of encoder hidden dim."
    )
    parser.add_argument(
        "--decoder_n_hidden_layers", type=int, default=8, help="Number of decoder hidden layers."
    )
    parser.add_argument(
        "--nl", type=str, default="softplus", help="Type of non-linearity: sine | relu."
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=0,
        help="Number of elements in the latent vector. Use 0 for reconstruction.",
    )
    parser.add_argument(
        "--sphere_init_params",
        nargs="+",
        type=float,
        default=[1.6, 1.0],
        help="Radius and scaling.",
    )
    parser.add_argument("--neuron_type", type=str, default="quadratic", help="Type of neuron.")
    parser.add_argument(
        "--encoder_type", type=str, default="none", help="Type of encoder: none | pointnet."
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="siren",
        help="Loss type to use: SPIN: igr[_wo_eik]_w_heat, StEik: siren[_wo_n]_w_div, siren[_wo_n], igr[_wo_n]",
    )
    parser.add_argument(
        "--div_decay",
        type=str,
        default="linear",
        help="Divergence term coefficient decay schedule: none | step | linear.",
    )
    parser.add_argument(
        "--div_decay_params",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 0.75],
        help="Decay schedule for divergence term coefficient. Not effective if div_decay = False. Format: [start, (location, value)*, end]",
    )
    parser.add_argument(
        "--div_type",
        type=str,
        default="dir_l1",
        help="Divergence term norm: dir_l1 | dir_l2 | full_l1 | full_l2.",
    )
    parser.add_argument("--grid_res", type=int, default=128, help="Uniform grid resolution.")
    parser.add_argument(
        "--nonmnfld_sample_type",
        type=str,
        default="uniform",
        help="How to sample points off the manifold. Currently supported sample types: grid | central_gaussian | grid_central_gaussian | uniform_central_gaussian.",
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default="mfgi",
        help="Initialization type: siren | geometric_sine | geometric_relu | mfgi.",
    )
    parser.add_argument(
        "--loss_weights",
        nargs="+",
        type=float,
        default=[2e4, 1e2, 1e2, 5e1, 1e2, 8e2],
        help="Loss terms weights: sdf | inter | normal | eikonal | div | heat.",
    )
    parser.add_argument(
        "--heat_lambda", type=float, default=30, help="Heat loss weight for eikonal loss."
    )
    parser.add_argument(
        "--heat_decay",
        type=str,
        default=None,
        help="Heat coefficient decay schedule: none | step | linear.",
    )
    parser.add_argument(
        "--heat_decay_params",
        nargs="+",
        type=float,
        default=[],
        help="Decay schedule for heat coefficient. Not effective if heat_decay = False. Format: [start, (location, value)*, end]",
    )
    parser.add_argument(
        "--heat_lambda_decay",
        type=str,
        default=None,
        help="Heat lambda decay schedule: none | step | linear.",
    )
    parser.add_argument(
        "--heat_lambda_decay_params",
        nargs="+",
        type=float,
        default=[],
        help="Decay schedule for heat lambda. Not effective if heat_lambda_decay = False. Format: [start, (location, value)*, end]",
    )

    # Sampling
    parser.add_argument(
        "--grid_range", type=float, default=1.2, help="Range of the grid to sample points."
    )
    parser.add_argument(
        "--nonmnfld_sample_std2",
        type=float,
        default=0.09,
        help="Standard deviation of the gaussian distribution to sample points off the manifold.",
    )
    parser.add_argument(
        "--n_random_samples", type=int, default=4096, help="Number of random samples."
    )

    # Misc
    parser.add_argument(
        "--model_dir", type=str, default="./models", help="Path to model directory for backup."
    )

    args = parser.parse_args()
    return args


def get_test_args():
    parser = configargparse.ArgumentParser(description="Local implicit functions test experiment.")
    parser.add_argument("--gpu_idx", type=int, default=1, help="Set < 0 to use CPU.")
    parser.add_argument("--logdir", type=str, default="./logs/", help="Log directory.")
    parser.add_argument("--file_name", type=str, default="daratech.ply", help="Trained model name.")
    parser.add_argument(
        "--n_points",
        type=int,
        default=0,
        help="Number of points in each point cloud. If 0, use training options.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Number of samples in a minibatch. If 0, use training.",
    )
    parser.add_argument(
        "--grid_res", type=int, default=512, help="Grid resolution for reconstruction."
    )
    parser.add_argument(
        "--export_mesh", type=bool, default=True, help="Indicator to export mesh as ply file."
    )
    parser.add_argument("--data_dir", type=str, default="", help="Path to dataset folder.")
    parser.add_argument(
        "--epoch_n",
        type=int,
        nargs="+",
        default=np.arange(0, 10000, 100).tolist(),
        help="Epoch number to evaluate.",
    )

    test_opt = parser.parse_args()
    test_opt.logdir = os.path.join(test_opt.logdir, test_opt.file_name.split(".")[0])
    param_filename = os.path.join(test_opt.logdir, "trained_models/params.pth")
    train_opt = torch.load(param_filename, weights_only=False)

    (
        test_opt.nl,
        test_opt.latent_size,
        test_opt.encoder_type,
        test_opt.n_iterations,
        test_opt.seed,
        test_opt.decoder_hidden_dim,
        test_opt.encoder_hidden_dim,
        test_opt.decoder_n_hidden_layers,
        test_opt.init_type,
        test_opt.neuron_type,
        test_opt.heat_lambda,
        test_opt.grid_range,
    ) = (
        train_opt.nl,
        train_opt.latent_size,
        train_opt.encoder_type,
        train_opt.n_iterations,
        train_opt.seed,
        train_opt.decoder_hidden_dim,
        train_opt.encoder_hidden_dim,
        train_opt.decoder_n_hidden_layers,
        train_opt.init_type,
        train_opt.neuron_type,
        train_opt.heat_lambda,
        train_opt.grid_range,
    )
    test_opt.n_point_total = train_opt.n_points

    if test_opt.n_points == 0:
        test_opt.n_points = train_opt.n_points
    if "parallel" in train_opt:
        test_opt.parallel = train_opt.parallel
    else:
        test_opt.parallel = False
    return test_opt
