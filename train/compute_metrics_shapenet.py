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
import point_cloud_utils as pcu

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
gt_meshes_path = args.gt_meshes_dir

print(f"dataset_path: {dataset_path}")
print(f"raw_dataset_path: {raw_dataset_path}")
print(f"mesh_path: {mesh_path}")
print(f"gt_meshes_dir: {gt_meshes_path}")

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
    "car",
    "chair",
    "airplane",
    "display",
    "table",
    "rifle",
    "cabinet",
    "loudspeaker",
    "telephone",
    "bench",
    "sofa",
    "watercraft",
    "lamp",
]
order = sorted(order)

IoUs = {}
chamfers = {}
hausdorffs = {}
RMSEs = {}
MAEs = {}
SMAPEs = {}
RMSEs_near_surface = {}
MAEs_near_surface = {}
SMAPEs_near_surface = {}
# for shape_class in os.listdir(mesh_path):
for shape_class in order:
    if shape_class not in os.listdir(mesh_path):
        continue
    shape_class_id = shape_class_name2id[shape_class]
    gt_shape_class_path = os.path.join(dataset_path, shape_class)
    gt_raw_shape_class_path = os.path.join(raw_dataset_path, shape_class_id)
    gt_mesh_shape_class_path = os.path.join(gt_meshes_path, shape_class_id)

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
    SMAPEs[shape_class] = []

    RMSEs_near_surface[shape_class] = []
    MAEs_near_surface[shape_class] = []
    SMAPEs_near_surface[shape_class] = []

    for shape_file in shape_files:
        shape = shape_file.split("/")[-2]
        print(f"Computing metrics for {shape_class}|{shape}")
        recon_shape_path = os.path.join(mesh_path, shape_class, shape_file)
        recon_mesh = trimesh.load(recon_shape_path)

        gt_pointcloud_path = os.path.join(gt_shape_class_path, shape + ".ply")
        gt_shape_weights_path = os.path.join(
            saved_weights_class_path, shape, "trained_models", "model.pth"
        )
        gt_pcd = trimesh.load(gt_pointcloud_path)
        gt_raw_shape_path = os.path.join(gt_raw_shape_class_path, shape)
        gt_mesh_path = os.path.join(gt_mesh_shape_class_path, shape + ".ply")
        points = np.load(
            os.path.join(gt_raw_shape_path, "points.npz")
        )  # [('points', (100000, 3)), ('occupancies', (12500,)), ('loc', (3,)), ('scale', ())]
        pointcloud = np.load(
            os.path.join(gt_raw_shape_path, "pointcloud.npz")
        )  # [('points', (100000, 3)), ('normals', (100000, 3)), ('loc', (3,)), ('scale', ())]
        gen_points = points["points"]  # shape: (100000,3), range: [-0.55, 0.55]^3
        occupancies = np.unpackbits(points["occupancies"])  # (100000)

        n_points = 15000
        n_samples = 10000
        grid_res = 512
        test_set = shape_3d.ReconDataset(
            gt_pointcloud_path,
            n_points=n_points,
            n_samples=1,
            grid_res=grid_res,
            sample_type="grid",
            requires_dist=False,
            grid_range=1.1,
            scale_method=args.pcd_scale_method,
        )
        default_cp, default_scale = test_set.get_cp_and_scale(scale_method="default")
        # print(f"Default cp: {default_cp}, Default scale: {default_scale}")
        # print(f"Default scale: {default_scale}")
        cp, scale = test_set.cp, test_set.scale
        # print(f"cp: {cp}, scale: {scale}")
        # print(f"scale: {scale}")
        # print()

        model.load_state_dict(
            torch.load(gt_shape_weights_path, map_location=device, weights_only=True)
        )
        model.to(device)

        # Compute surface metrics
        eval_points_np = (gen_points - cp) / scale
        eval_points = torch.tensor(eval_points_np, device=device, dtype=torch.float32)
        with torch.no_grad():
            res = model.decoder(eval_points)

        pred_occupancies = (res.reshape(-1) < 0).int().detach().cpu().numpy()
        iou = (occupancies & pred_occupancies).sum() / (occupancies | pred_occupancies).sum()
        IoUs[shape_class].append(iou)

        gt_points = gt_pcd.vertices
        recon_points = trimesh.sample.sample_surface(recon_mesh, 30000)[0]
        chamfer_dist, hausdorff_dist, cd_re2gt, cd_gt2re, hd_re2gt, hd_gt2re = compute_dists(
            recon_points, gt_points
        )
        chamfers[shape_class].append(chamfer_dist)
        hausdorffs[shape_class].append(hausdorff_dist)

        # Compute distance metrics
        gen_points_unit = gen_points / np.max(gen_points) # Transform the uniforms samples to [-1,1]^3
        eval_points_dists_np = gen_points_unit * default_scale / scale # If the predicted SDF is spatially scaled by (scale / default_scale), it will correspond to StEik visualization space
        eval_points_dists = torch.tensor(eval_points_dists_np, device=device, dtype=torch.float32)
        with torch.no_grad():
            res_dists = model.decoder(eval_points_dists)

        dists_pred = res_dists.detach().cpu().numpy()
        dists_pred = dists_pred * scale / default_scale # The SDF values should be scaled by (scale / default_scale) to be in the same space as the StEik visualization space

        vm, fm = pcu.load_mesh_vf(gt_mesh_path, dtype=np.float32)
        # vm, fm = pcu.make_mesh_watertight(v, f, 20000)
        vm = vm.astype(np.float32)

        dists_gt, _, _ = pcu.signed_distance_to_mesh(
            (gen_points_unit * default_scale + cp).astype(np.float32), vm, fm
        ) # "(points - cp) / default_scale" will transform the mesh/points to the visualization space, so take the inverse of that to transform the evaluation points to the mesh space
        dists_gt = dists_gt[..., None]
        dists_gt = dists_gt / default_scale

        # # use plotly to visualize dists_gt on gen_points
        # import plotly.graph_objects as go

        # fig = go.Figure()

        # fig.add_trace(go.Scatter3d(x=gen_points_unit[..., 0], y=gen_points_unit[..., 1], z=gen_points_unit[..., 2], mode='markers', marker=dict(size=1, color=dists_gt[..., 0], colorscale='RdBu_r', opacity=0.8, cmin=-1, cmax=1), name='gt'))

        # fig.add_trace(go.Scatter3d(x=gen_points_unit[..., 0], y=gen_points_unit[..., 1], z=gen_points_unit[..., 2], mode='markers', marker=dict(size=1, color=dists_pred[..., 0], colorscale='RdBu_r', opacity=0.8, cmin=-1, cmax=1), name='pred'))

        # dists_gt_occupancies = dists_gt < 0
        # dists_gt_occupancies_mask = dists_gt_occupancies[..., 0]
        # dists_gt_occupancies_points = gen_points_unit[dists_gt_occupancies_mask]

        # dists_pred_occupancies = dists_pred < 0
        # dists_pred_occupancies_mask = dists_pred_occupancies[..., 0]
        # dists_pred_occupancies_points = gen_points_unit[dists_pred_occupancies_mask]

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=dists_gt_occupancies_points[..., 0],
        #         y=dists_gt_occupancies_points[..., 1],
        #         z=dists_gt_occupancies_points[..., 2],
        #         mode="markers",
        #         marker=dict(size=1, color="blue", opacity=0.5),
        #     )
        # )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=dists_pred_occupancies_points[..., 0],
        #         y=dists_pred_occupancies_points[..., 1],
        #         z=dists_pred_occupancies_points[..., 2],
        #         mode="markers",
        #         marker=dict(size=1, color="red", opacity=0.5),
        #     )
        # )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=gen_points_unit[..., 0],
        #         y=gen_points_unit[..., 1],
        #         z=gen_points_unit[..., 2],
        #         mode="markers",
        #         marker=dict(size=1, color="black", opacity=0.1),
        #     )
        # )

        # fig.write_html("dists_compare.html")

        # fig = go.Figure()

        # # Visualize gt mesh occupancy
        # gt_mesh_occupancies = dists_gt < 0
        # gt_mesh_occupancies_mask = gt_mesh_occupancies[..., 0]
        # gt_mesh_occupancies_points = eval_points_np[gt_mesh_occupancies_mask]
        # gt_mesh_nonoccupancies_points = eval_points_np[~gt_mesh_occupancies_mask]
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=gt_mesh_occupancies_points[..., 0],
        #         y=gt_mesh_occupancies_points[..., 1],
        #         z=gt_mesh_occupancies_points[..., 2],
        #         mode="markers",
        #         marker=dict(size=1, color="blue", opacity=0.5),
        #     )
        # )

        # occupancies_mask = occupancies == 1
        # occupancies_points = eval_points_np[occupancies_mask]
        # nonoccupancies_points = eval_points_np[~occupancies_mask]

        # pred_occupancies_mask = pred_occupancies == 1
        # pred_occupancies_points = eval_points_np[pred_occupancies_mask]
        # pred_nonoccupancies_points = eval_points_np[~pred_occupancies_mask]

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=occupancies_points[..., 0],
        #         y=occupancies_points[..., 1],
        #         z=occupancies_points[..., 2],
        #         mode="markers",
        #         marker=dict(size=1, color="green", opacity=0.5),
        #     )
        # )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=pred_occupancies_points[..., 0],
        #         y=pred_occupancies_points[..., 1],
        #         z=pred_occupancies_points[..., 2],
        #         mode="markers",
        #         marker=dict(size=1, color="red", opacity=0.5),
        #     )
        # )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=eval_points_np[..., 0],
        #         y=eval_points_np[..., 1],
        #         z=eval_points_np[..., 2],
        #         mode="markers",
        #         marker=dict(size=1, color="black", opacity=0.1),
        #     )
        # )
        # fig.write_html("occupancies_compare.html")

        # exit(0)

        rmse = np.sqrt(np.mean((dists_gt - dists_pred) ** 2))
        mae = np.mean(np.abs(dists_gt - dists_pred))
        smape = 2 * np.mean(np.abs(dists_gt - dists_pred) / (np.abs(dists_gt) + np.abs(dists_pred)))

        # compute these 4 metrics for points where |dists_gt| < threshold
        threshold = 0.1
        where_gt = np.abs(dists_gt) < threshold
        rmse_near_surface = np.sqrt(np.mean((dists_gt[where_gt] - dists_pred[where_gt]) ** 2))
        mae_near_surface = np.mean(np.abs(dists_gt[where_gt] - dists_pred[where_gt]))
        smape_near_surface = 2 * np.mean(np.abs(dists_gt[where_gt] - dists_pred[where_gt]) / (np.abs(dists_gt[where_gt]) + np.abs(dists_pred[where_gt])))

        RMSEs[shape_class].append(rmse)
        MAEs[shape_class].append(mae)
        SMAPEs[shape_class].append(smape)

        RMSEs_near_surface[shape_class].append(rmse_near_surface)
        MAEs_near_surface[shape_class].append(mae_near_surface)
        SMAPEs_near_surface[shape_class].append(smape_near_surface)

        logging.info(
            f"{shape_class}|{shape}: IoU = {iou:.4f}, Chamfer = {chamfer_dist:.4f}, Hausdorff = {hausdorff_dist:.4f}; RMSE = {rmse:.4f}, MAE = {mae:.4f}, SMAPE = {smape:.4f}; RMSE_near_surface = {rmse_near_surface:.4f}, MAE_near_surface = {mae_near_surface:.4f}, SMAPE_near_surface = {smape_near_surface:.4f}"
        )
        logging.info("")

for shape_class in order:
    if shape_class not in IoUs:
        continue
    # Log metrics for each shape class
    IoUs_class = np.array(IoUs[shape_class])
    chamfer_distances_class = np.array(chamfers[shape_class])
    hausdorff_distances_class = np.array(hausdorffs[shape_class])
    # clear nan and inf values
    RMSEs_class = np.array(RMSEs[shape_class])
    RMSEs_class = RMSEs_class[np.isfinite(RMSEs_class)]
    MAEs_class = np.array(MAEs[shape_class])
    MAEs_class = MAEs_class[np.isfinite(MAEs_class)]
    SMAPEs_class = np.array(SMAPEs[shape_class])
    SMAPEs_class = SMAPEs_class[np.isfinite(SMAPEs_class)]

    RMSEs_near_surface_class = np.array(RMSEs_near_surface[shape_class])
    RMSEs_near_surface_class = RMSEs_near_surface_class[np.isfinite(RMSEs_near_surface_class)]
    MAEs_near_surface_class = np.array(MAEs_near_surface[shape_class])
    MAEs_near_surface_class = MAEs_near_surface_class[np.isfinite(MAEs_near_surface_class)]
    SMAPEs_near_surface_class = np.array(SMAPEs_near_surface[shape_class])
    SMAPEs_near_surface_class = SMAPEs_near_surface_class[np.isfinite(SMAPEs_near_surface_class)]

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
        f"SMAPE (mean/median/std): {np.mean(SMAPEs_class):.4f}/{np.median(SMAPEs_class):.4f}/{np.std(SMAPEs_class):.4f}"
    )
    logging.info(
        f"RMSE_near_surface (mean/median/std): {np.mean(RMSEs_near_surface_class):.4f}/{np.median(RMSEs_near_surface_class):.4f}/{np.std(RMSEs_near_surface_class):.4f}"
    )
    logging.info(
        f"MAE_near_surface (mean/median/std): {np.mean(MAEs_near_surface_class):.4f}/{np.median(MAEs_near_surface_class):.4f}/{np.std(MAEs_near_surface_class):.4f}"
    )
    logging.info(
        f"SMAPE_near_surface (mean/median/std): {np.mean(SMAPEs_near_surface_class):.4f}/{np.median(SMAPEs_near_surface_class):.4f}/{np.std(SMAPEs_near_surface_class):.4f}"
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
SMAPEs = np.array([item for sublist in SMAPEs.values() for item in sublist])
SMAPEs = SMAPEs[np.isfinite(SMAPEs)]

RMSEs_near_surface = np.array([item for sublist in RMSEs_near_surface.values() for item in sublist])
RMSEs_near_surface = RMSEs_near_surface[np.isfinite(RMSEs_near_surface)]
MAEs_near_surface = np.array([item for sublist in MAEs_near_surface.values() for item in sublist])
MAEs_near_surface = MAEs_near_surface[np.isfinite(MAEs_near_surface)]
SMAPEs_near_surface = np.array(
    [item for sublist in SMAPEs_near_surface.values() for item in sublist]
)
SMAPEs_near_surface = SMAPEs_near_surface[np.isfinite(SMAPEs_near_surface)]

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
    f"SMAPE (mean/median/std): {np.mean(SMAPEs):.4f}/{np.median(SMAPEs):.4f}/{np.std(SMAPEs):.4f}"
)
logging.info(
    f"RMSE_near_surface (mean/median/std): {np.mean(RMSEs_near_surface):.4f}/{np.median(RMSEs_near_surface):.4f}/{np.std(RMSEs_near_surface):.4f}"
)
logging.info(
    f"MAE_near_surface (mean/median/std): {np.mean(MAEs_near_surface):.4f}/{np.median(MAEs_near_surface):.4f}/{np.std(MAEs_near_surface):.4f}"
)
logging.info(
    f"SMAPE_near_surface (mean/median/std): {np.mean(SMAPEs_near_surface):.4f}/{np.median(SMAPEs_near_surface):.4f}/{np.std(SMAPEs_near_surface):.4f}"
)
