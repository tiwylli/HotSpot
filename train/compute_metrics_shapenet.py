# This file is borrowed from DiGS: https://github.com/Chumbyte/DiGS
import numpy as np
import json
import os
import trimesh
from scipy.spatial import cKDTree as KDTree
import sys
import glob

import trimesh.proximity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import parser

import torch
from dataset import shape_3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.Net as model
import logging
from pysdf import SDF

device = torch.device("cuda")
args = parser.get_train_args()
model = model.Network(
    latent_size=args.latent_size,
    in_dim=3,
    decoder_hidden_dim=args.decoder_hidden_dim,
    nl=args.nl,
    encoder_type=args.encoder_type,
    decoder_n_hidden_layers=args.decoder_n_hidden_layers,
    neuron_type=args.neuron_type,
    init_type=args.init_type,
    sphere_init_params=args.sphere_init_params,
    n_repeat_period=args.n_repeat_period,
)


dataset_path = args.data_dir
raw_dataset_path = args.raw_dataset_path
mesh_path = args.log_dir

print(dataset_path)
print(raw_dataset_path)
print(mesh_path)

# Print metrics to file in logdir as well
out_path = os.path.join(args.log_dir, "metric_summary.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(out_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.info(f"Computing metrics in {mesh_path}")

shape_class_name_dict = {
    "04256520": "sofa",
    "02691156": "airplane",
    "03636649": "lamp",
    "04401088": "telephone",
    "04530566": "watercraft",
    "03691459": "loudspeaker",
    "03001627": "chair",
    "02933112": "cabinet",
    "04379243": "table",
    "03211117": "display",
    "02958343": "car",
    "02828884": "bench",
    "04090263": "rifle",
}

shape_class_name2id = {v: k for k, v in shape_class_name_dict.items()}


def compute_dists(recon_points, gt_points, eval_type="Default"):
    recon_kd_tree = KDTree(recon_points)
    gt_kd_tree = KDTree(gt_points)
    re2gt_distances, re2gt_vertex_ids = recon_kd_tree.query(gt_points, workers=4)
    gt2re_distances, gt2re_vertex_ids = gt_kd_tree.query(recon_points, workers=4)
    if eval_type == "DeepSDF":
        cd_re2gt = np.mean(re2gt_distances**2)
        cd_gt2re = np.mean(gt2re_distances**2)
        hd_re2gt = np.max(re2gt_distances)
        hd_gt2re = np.max(gt2re_distances)
        chamfer_dist = cd_re2gt + cd_gt2re
        hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
    else:
        cd_re2gt = np.mean(re2gt_distances)
        cd_gt2re = np.mean(gt2re_distances)
        hd_re2gt = np.max(re2gt_distances)
        hd_gt2re = np.max(gt2re_distances)
        chamfer_dist = 0.5 * (cd_re2gt + cd_gt2re)
        hausdorff_distance = np.max((hd_re2gt, hd_gt2re))
    return chamfer_dist, hausdorff_distance, cd_re2gt, cd_gt2re, hd_re2gt, hd_gt2re


order = [
    # "car",
    # "chair",
    # "airplane",
    # "display",
    # "table",
    # "rifle",
    # "cabinet",
    # "loudspeaker",
    # "telephone",
    "bench",
    # "sofa",
    # "watercraft",
    # "lamp",
]
order = sorted(order)

IoUs = {}
chamfers = {}
hausdorffs = {}
RMSEs = {}
MAEs = {}
MAPEs = {}
SMAPEs = {}
# for shape_class in os.listdir(mesh_path):
for shape_class in order:
    if shape_class not in os.listdir(mesh_path):
        continue
    shape_class_id = shape_class_name2id[shape_class]
    gt_shape_class_path = os.path.join(dataset_path, shape_class)
    gt_raw_shape_class_path = os.path.join(raw_dataset_path, shape_class_id)

    result_meshes_path_list = glob.glob(os.path.join(mesh_path, shape_class, "*", "output.ply"))
    saved_weights_class_path = os.path.join(mesh_path, shape_class)
    shape_files = [
        os.path.join(*os.path.split(f)[-2:]) for f in result_meshes_path_list if "_iter_" not in f
    ]
    print("Found {} files for {}".format(len(shape_files), shape_class))

    IoUs[shape_class] = []
    chamfers[shape_class] = []
    hausdorffs[shape_class] = []
    RMSEs[shape_class] = []
    MAEs[shape_class] = []
    MAPEs[shape_class] = []
    SMAPEs[shape_class] = []
    for shape_file in shape_files:
        shape = shape_file.split("/")[-2]
        print(f"Computing metrics for {shape_class}|{shape}")
        recon_shape_path = os.path.join(mesh_path, shape_class, shape_file)
        recon_mesh = trimesh.load(recon_shape_path)

        gt_shape_path = os.path.join(gt_shape_class_path, shape + ".ply")
        gt_shape_weights_path = os.path.join(
            saved_weights_class_path, shape, "trained_models", "model.pth"
        )
        gt_pc = trimesh.load(gt_shape_path)
        gt_raw_shape_path = os.path.join(gt_raw_shape_class_path, shape)
        points = np.load(
            os.path.join(gt_raw_shape_path, "points.npz")
        )  # [('points', (100000, 3)), ('occupancies', (12500,)), ('loc', (3,)), ('scale', ())]
        pointcloud = np.load(
            os.path.join(gt_raw_shape_path, "pointcloud.npz")
        )  # [('points', (100000, 3)), ('normals', (100000, 3)), ('loc', (3,)), ('scale', ())]
        gen_points = points["points"]  # (100000,3)
        occupancies = np.unpackbits(points["occupancies"])  # (100000)

        n_points = 15000
        n_samples = 10000
        grid_res = 512
        test_set = shape_3d.ReconDataset(
            gt_shape_path,
            n_points=n_points,
            n_samples=1,
            grid_res=grid_res,
            sample_type="grid",
            requires_dist=False,
            grid_range=1.1,
        )
        cp, scale, bbox = test_set.cp, test_set.scale, test_set.bbox
        model.load_state_dict(
            torch.load(gt_shape_weights_path, map_location=device, weights_only=True)
        )
        model.to(device)

        eval_points_np = (gen_points - cp) / scale
        eval_points = torch.tensor(eval_points_np, device=device, dtype=torch.float32)
        res = model.decoder(eval_points)

        pred_occupancies = (res.reshape(-1) < 0).int().detach().cpu().numpy()
        iou = (occupancies & pred_occupancies).sum() / (occupancies | pred_occupancies).sum()
        IoUs[shape_class].append(iou)

        gt_points = gt_pc.vertices
        recon_points = trimesh.sample.sample_surface(recon_mesh, 30000)[0]
        chamfer_dist, hausdorff_dist, cd_re2gt, cd_gt2re, hd_re2gt, hd_gt2re = compute_dists(
            recon_points, gt_points
        )
        chamfers[shape_class].append(chamfer_dist)
        hausdorffs[shape_class].append(hausdorff_dist)

        # dists_pred = res.detach().cpu().numpy()
        # f = SDF(gt_pc.vertices)
        # dists_gt = f(eval_points_np)[..., None] # (100000, 1)
        # dists_gt = np.where(occupancies[:, None] == 0, np.abs(dists_gt), -np.abs(dists_gt))

        # # use plotly to visualize dists_gt on (gen_points - cp) / scale in 3D
        # import plotly.graph_objects as go
        # fig = go.Figure(data=[go.Scatter3d(x=eval_points_np[..., 0], y=eval_points_np[..., 1], z=eval_points_np[..., 2], mode='markers', marker=dict(size=1, color=dists_gt[..., 0], colorscale='RdBu_r', opacity=0.8, cmin=-3, cmax=3))])
        # # save as html
        # fig.write_html("gt_dist.html")
        
        # fig = go.Figure(data=[go.Scatter3d(x=eval_points_np[..., 0], y=eval_points_np[..., 1], z=eval_points_np[..., 2], mode='markers', marker=dict(size=1, color=dists_pred[..., 0], colorscale='RdBu_r', opacity=0.8, cmin=-3, cmax=3))])
        # fig.write_html("pred_dist.html")

        # occupancies_mask = occupancies == 1
        # occupancies_points = eval_points_np[occupancies_mask]

        # pred_occupancies_mask = pred_occupancies == 1
        # pred_occupancies_points = eval_points_np[pred_occupancies_mask]

        # fig = go.Figure()

        # fig.add_trace(go.Scatter3d(
        #     x=occupancies_points[..., 0],
        #     y=occupancies_points[..., 1],
        #     z=occupancies_points[..., 2],
        #     mode='markers',
        #     marker=dict(size=1, color='blue', opacity=0.5)
        # ))

        # fig.add_trace(go.Scatter3d(
        #     x=pred_occupancies_points[..., 0],
        #     y=pred_occupancies_points[..., 1],
        #     z=pred_occupancies_points[..., 2],
        #     mode='markers',
        #     marker=dict(size=1, color='red', opacity=0.5)
        # ))
        # fig.write_html("occupancies_compare.html")

        # exit(0)

        # rmse = np.sqrt(np.mean((dists_gt - dists_pred) ** 2))
        # mae = np.mean(np.abs(dists_gt - dists_pred))
        # mape = np.mean(np.abs(dists_gt - dists_pred) / np.abs(dists_gt))
        # smape = 2 * np.mean(np.abs(dists_gt - dists_pred) / (np.abs(dists_gt) + np.abs(dists_pred)))
        rmse = 0
        mae = 0
        mape = 0
        smape = 0

        RMSEs[shape_class].append(rmse)
        MAEs[shape_class].append(mae)
        MAPEs[shape_class].append(mape)
        SMAPEs[shape_class].append(smape)

        logging.info(
            f"{shape_class}|{shape}: IoU = {iou:.4f}, Chamfer = {chamfer_dist:.4f}, Hausdorff = {hausdorff_dist:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.4f}, SMAPE = {smape:.4f}"
        )
        logging.info("")

for shape_class in order:
    # Log metrics for each shape class
    IoUs_class = np.array(IoUs[shape_class])
    chamfer_distances_class = np.array(chamfers[shape_class])
    hausdorff_distances_class = np.array(hausdorffs[shape_class])
    # clear nan and inf values
    RMSEs_class = np.array(RMSEs[shape_class])
    RMSEs_class = RMSEs_class[np.isfinite(RMSEs_class)]
    MAEs_class = np.array(MAEs[shape_class])
    MAEs_class = MAEs_class[np.isfinite(MAEs_class)]
    MAPEs_class = np.array(MAPEs[shape_class])
    MAPEs_class = MAPEs_class[np.isfinite(MAPEs_class)]
    SMAPEs_class = np.array(SMAPEs[shape_class])
    SMAPEs_class = SMAPEs_class[np.isfinite(SMAPEs_class)]

    logging.info(f"Shape class: {shape_class}")
    logging.info("==========================================")
    logging.info(
        f"IoU (mean/median/std): {IoUs_class.mean():.4f}/{np.median(IoUs_class):.4f}/{IoUs_class.std():.4f}"
    )
    logging.info(
        f"Chamfer (mean/median/std): {chamfer_distances_class.mean():.4f}/{np.median(chamfer_distances_class):.4f}/{chamfer_distances_class.std():.4f}"
    )
    logging.info(
        f"Hausdorff (mean/median/std): {hausdorff_distances_class.mean():.4f}/{np.median(hausdorff_distances_class):.4f}/{hausdorff_distances_class.std():.4f}"
    )
    logging.info(
        f"RMSE (mean/median/std): {np.mean(RMSEs_class):.4f}/{np.median(RMSEs_class):.4f}/{np.std(RMSEs_class):.4f}"
    )
    logging.info(
        f"MAE (mean/median/std): {np.mean(MAEs_class):.4f}/{np.median(MAEs_class):.4f}/{np.std(MAEs_class):.4f}"
    )
    logging.info(
        f"MAPE (mean/median/std): {np.mean(MAPEs_class):.4f}/{np.median(MAPEs_class):.4f}/{np.std(MAPEs_class):.4f}"
    )
    logging.info(
        f"SMAPE (mean/median/std): {np.mean(SMAPEs_class):.4f}/{np.median(SMAPEs_class):.4f}/{np.std(SMAPEs_class):.4f}"
    )
    logging.info("")

# Log metrics for all shape classes
IoUs = np.array([item for sublist in IoUs.values() for item in sublist])
chamfer_distances = np.array([item for sublist in chamfers.values() for item in sublist])
hausdorff_distances = np.array([item for sublist in hausdorffs.values() for item in sublist])
# clear nan and inf values
RMSEs = np.array([item for sublist in RMSEs.values() for item in sublist])
RMSEs = RMSEs[np.isfinite(RMSEs)]
MAEs = np.array([item for sublist in MAEs.values() for item in sublist])
MAEs = MAEs[np.isfinite(MAEs)]
MAPEs = np.array([item for sublist in MAPEs.values() for item in sublist])
MAPEs = MAPEs[np.isfinite(MAPEs)]
SMAPEs = np.array([item for sublist in SMAPEs.values() for item in sublist])
SMAPEs = SMAPEs[np.isfinite(SMAPEs)]

logging.info("All shape classes")
logging.info("==========================================")
logging.info(f"IoU (mean/median/std): {IoUs.mean():.4f}/{np.median(IoUs):.4f}/{IoUs.std():.4f}")
logging.info(
    f"Chamfer (mean/median/std): {chamfer_distances.mean():.4f}/{np.median(chamfer_distances):.4f}/{chamfer_distances.std():.4f}"
)
logging.info(
    f"Hausdorff (mean/median/std): {hausdorff_distances.mean():.4f}/{np.median(hausdorff_distances):.4f}/{hausdorff_distances.std():.4f}"
)
logging.info(
    f"RMSE (mean/median/std): {np.mean(RMSEs):.4f}/{np.median(RMSEs):.4f}/{np.std(RMSEs):.4f}"
)
logging.info(f"MAE (mean/median/std): {np.mean(MAEs):.4f}/{np.median(MAEs):.4f}/{np.std(MAEs):.4f}")
logging.info(
    f"MAPE (mean/median/std): {np.mean(MAPEs):.4f}/{np.median(MAPEs):.4f}/{np.std(MAPEs):.4f}"
)
logging.info(
    f"SMAPE (mean/median/std): {np.mean(SMAPEs):.4f}/{np.median(SMAPEs):.4f}/{np.std(SMAPEs):.4f}"
)
