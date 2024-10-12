import logging
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree
from skimage import measure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataset.shape_2d as dataset
import models.Net as model
import utils.parser as parser
import plotly.graph_objects as go


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


if __name__ == "__main__":
    device = torch.device("cuda")
    args = parser.get_train_args()
    args.vis_grid_res = 512 # hardcode vis_grid_res to 512
    assert args.task == "2d", "This script is only for 2D shapes"
    exp_path = args.log_dir
    out_path = os.path.join(exp_path, "metric_summary.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(out_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Computing metrics in {exp_path}")

    model = model.Network(
        latent_size=args.latent_size,
        in_dim=2,
        decoder_hidden_dim=args.decoder_hidden_dim,
        nl=args.nl,
        encoder_type=args.encoder_type,
        decoder_n_hidden_layers=args.decoder_n_hidden_layers,
        init_type=args.init_type,
        neuron_type=args.neuron_type,
        sphere_init_params=args.sphere_init_params,
    )

    IoUs = []
    chamfer_distances = []
    hausdorff_distances = []
    RMSEs = []
    MAEs = []
    MAPEs = []
    SMAPEs = []

    shape_names = [
        "circle",
        "L",
        "square",
        "snowflake",
        "starhex",
        "button",
        "target",
        "bearing",
        "snake",
        "seaurchin",
        "peace",
        "boomerangs",
        "fragments",
        "house",
    ]
    shape_names = sorted(shape_names)

    # Initialize visualization grid
    x, y = np.linspace(-args.vis_grid_range, args.vis_grid_range, args.vis_grid_res), np.linspace(
        -args.vis_grid_range, args.vis_grid_range, args.vis_grid_res
    )
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()
    vis_grid_points = np.stack([xx, yy], axis=-1)
    vis_grid_points = vis_grid_points[None, ...]  # (1, grid_res * grid_res, dim)
    vis_grid_points = torch.tensor(vis_grid_points, dtype=torch.float32).to(device)

    for shape_name in shape_names:
        gt_shape_weights_path = os.path.join(exp_path, shape_name, "trained_models", "model.pth")

        test_set = dataset.get2D_dataset(
            n_points=args.n_points,
            n_samples=1,
            grid_res=args.grid_res,
            grid_range=args.grid_range,
            sample_type="grid",
            resample=True,
            shape_type=shape_name,
        )
        model.load_state_dict(
            torch.load(gt_shape_weights_path, map_location=device, weights_only=True)
        )
        model.to(device)

        # Compute IoU
        with torch.no_grad():
            dists_pred = model.decoder(vis_grid_points).cpu().numpy()
        occupancies_pred = (dists_pred.reshape(-1) < 0).astype(int)

        dists_gt, _ = test_set.get_points_distances_and_normals(
            vis_grid_points.squeeze().detach().cpu().numpy()
        )
        occupancies_gt = (dists_gt.reshape(-1) < 0).astype(int)

        iou = (occupancies_gt & occupancies_pred).sum() / (occupancies_gt | occupancies_pred).sum()

        # # Plot occupancies for debug
        # # Create an array containing the coords of the occupied points
        # occupied_pred = vis_grid_points[0, occupancies_pred == 1].cpu().numpy()
        # occupied_gt = vis_grid_points[0, occupancies_gt == 1].cpu().numpy()
        # fig = go.Figure()
        # fig.add_trace(
        #     go.Scatter(
        #         x=occupied_pred[:, 0],
        #         y=occupied_pred[:, 1],
        #         mode="markers",
        #         marker=dict(
        #             size=3,
        #             color="red",
        #             opacity=0.8,
        #             showscale=True,
        #         ),
        #     )
        # ) 
        # fig.add_trace(
        #     go.Scatter(
        #         x=occupied_gt[:, 0],
        #         y=occupied_gt[:, 1],
        #         mode="markers",
        #         marker=dict(
        #             size=3,
        #             color="blue",
        #             opacity=0.8,
        #             showscale=True,
        #         ),
        #     )
        # )
        # fig.show()
        # exit(0)

        # Compute Chamfer and Hausdorff distances
        gt_points = test_set.mnfld_points
        contours = measure.find_contours(
            dists_pred.reshape(args.vis_grid_res, args.vis_grid_res), level=0
        )
        contours = [contour / (args.vis_grid_res - 1) * 2 * args.vis_grid_range - args.vis_grid_range for contour in contours]
        pred_points = np.concatenate(contours, axis=0)
        pred_points = np.stack([pred_points[:, 1], pred_points[:, 0]], axis=-1)

        chamfer_dist, hausdorff_dist, cd_re2gt, cd_gt2re, hd_re2gt, hd_gt2re = compute_dists(
            pred_points, gt_points, eval_type="DeepSDF"
        )

        # # Visualize contour points
        # fig = go.Figure()
        # fig.add_trace(
        #     go.Scatter(
        #         x=pred_points[:, 0],
        #         y=pred_points[:, 1],
        #         mode="markers",
        #         marker=dict(
        #             size=3,
        #             color="blue",
        #             opacity=0.5,
        #             showscale=True,
        #         ),
        #     )
        # ) 
        # fig.add_trace(
        #     go.Scatter(
        #         x=gt_points[:, 0],
        #         y=gt_points[:, 1],
        #         mode="markers",
        #         marker=dict(
        #             size=3,
        #             color="green",
        #             opacity=0.5,
        #             showscale=True,
        #         ),
        #     )
        # )
        # fig.show()

        # Compute RMSE, MAE, MAPE, SMAPE
        rmse = np.sqrt(np.mean((dists_gt - dists_pred) ** 2))
        mae = np.mean(np.abs(dists_gt - dists_pred))
        mape = np.mean(np.abs(dists_gt - dists_pred) / np.abs(dists_gt))
        smape = 2 * np.mean(np.abs(dists_gt - dists_pred) / (np.abs(dists_gt) + np.abs(dists_pred)))

        IoUs.append(iou)
        chamfer_distances.append(chamfer_dist)
        hausdorff_distances.append(hausdorff_dist)
        RMSEs.append(rmse)
        MAEs.append(mae)
        MAPEs.append(mape)
        SMAPEs.append(smape)

        logging.info(
            f"{shape_name}: IoU = {iou:.4f}, Chamfer = {chamfer_dist:.4e}, Hausdorff = {hausdorff_dist:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, MAPE = {mape:.4f}, SMAPE = {smape:.4f}"
        )

    # Calculate mean, median, and std
    IoUs = np.array(IoUs)
    chamfer_distances = np.array(chamfer_distances)
    hausdorff_distances = np.array(hausdorff_distances)
    # clear nan and inf values
    MAPEs = np.array(MAPEs)
    MAPEs = MAPEs[np.isfinite(MAPEs)]
    SMAPEs = np.array(SMAPEs)
    SMAPEs = SMAPEs[np.isfinite(SMAPEs)]

    logging.info("")
    logging.info(f"IoU (mean/median/std): {IoUs.mean():.4f}/{np.median(IoUs):.4f}/{IoUs.std():.4f}")
    logging.info(
        f"Chamfer (mean/median/std): {chamfer_distances.mean():.4e}/{np.median(chamfer_distances):.4e}/{chamfer_distances.std():.4e}"
    )
    logging.info(
        f"Hausdorff (mean/median/std): {hausdorff_distances.mean():.4f}/{np.median(hausdorff_distances):.4f}/{hausdorff_distances.std():.4f}"
    )
    logging.info(
        f"RMSE (mean/median/std): {np.mean(RMSEs):.4f}/{np.median(RMSEs):.4f}/{np.std(RMSEs):.4f}"
    )
    logging.info(
        f"MAE (mean/median/std): {np.mean(MAEs):.4f}/{np.median(MAEs):.4f}/{np.std(MAEs):.4f}"
    )
    logging.info(
        f"MAPE (mean/median/std): {np.mean(MAPEs):.4f}/{np.median(MAPEs):.4f}/{np.std(MAPEs):.4f}"
    )
    logging.info(
        f"SMAPE (mean/median/std): {np.mean(SMAPEs):.4f}/{np.median(SMAPEs):.4f}/{np.std(SMAPEs):.4f}"
    )
