# This file is borrowed from DiGS: https://github.com/Chumbyte/DiGS
import numpy as np
import json
import os
import trimesh
from scipy.spatial import cKDTree as KDTree
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import parser
import logging

args = parser.get_train_args()

scan_path = os.path.join(args.data_dir, "scans")
gt_path = os.path.join(args.data_dir, "ground_truth")
mesh_path = args.log_dir

# Or set manually
# scan_path = '/home/chamin/DiGSGithub/DiBS/data/deep_geometric_prior_data/scans'
# gt_path = '/home/chamin/DiGSGithub/DiBS/data/deep_geometric_prior_data/ground_truth'
# mesh_path = '/home/chamin/DiGSGithub/DiBS/surface_reconstruction/log/surface_reconstruction2/DiGS_surf_recon_experiment/result_meshes'

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
logging.info("Metrics on SRB (DC, DH, DC-> DH->)")

eval_type = "Default"


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


shapes = ["anchor", "daratech", "dc", "gargoyle", "lord_quas"]
chamfer_dists = []
hausdorff_dists = []
sc_cd_re2gts = []
sc_hd_re2gts = []
#shapes = ["anchor"]
for shape in shapes:
    scan_shape_path = os.path.join(scan_path, "{}.ply".format(shape))
    gt_shape_path = os.path.join(gt_path, "{}.xyz".format(shape))
    recon_shape_path = os.path.join(mesh_path, shape, shape, "output.ply")
    # recon_shape_path = os.path.join(mesh_path, '{}_iter_6000.ply'.format(shape))
    # print(recon_shape_path)
    if not os.path.exists(recon_shape_path):
        continue
    scan_pc = trimesh.load(scan_shape_path)
    gt_pc = trimesh.load(gt_shape_path)
    recon_mesh = trimesh.load(recon_shape_path)

    scan_points = scan_pc.vertices
    gt_points = gt_pc.vertices
    # recon_points = trimesh.sample.sample_surface(recon_mesh, 30000)[0]
    recon_points = trimesh.sample.sample_surface(recon_mesh, 1000000)[0]
    chamfer_dist, hausdorff_distance, cd_re2gt, cd_gt2re, hd_re2gt, hd_gt2re = compute_dists(
        recon_points, gt_points
    )
    sc_chamfer_dist, sc_hausdorff_distance, sc_cd_re2gt, sc_cd_gt2re, sc_hd_re2gt, sc_hd_gt2re = (
        compute_dists(recon_points, scan_points)
    )
    logging.info(
        "{: <10} {:.3f} {:.3f} {:.3f} {:.3f}".format(
            shape, chamfer_dist, hausdorff_distance, sc_cd_re2gt, sc_hd_re2gt
        )
    )
    chamfer_dists.append(chamfer_dist)
    hausdorff_dists.append(hausdorff_distance)
    sc_cd_re2gts.append(sc_cd_re2gt)
    sc_hd_re2gts.append(sc_hd_re2gt)
    
logging.info(
    "{: <10} {:.3f} {:.3f} {:.3f} {:.3f}".format(
        "Average",
        np.mean(chamfer_dists),
        np.mean(hausdorff_dists),
        np.mean(sc_cd_re2gts),
        np.mean(sc_hd_re2gts),
    )
)
