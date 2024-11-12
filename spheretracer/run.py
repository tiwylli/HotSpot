import torch
import numpy  as np
import sys
import os

from ray_tracing import RayTracing
from pyhocon import ConfigFactory
from spheretracer_utils import rend_util, parser
from spheretracer_utils.eval_util import generate_eval_cams
from spheretracer_utils.save_results import save_map_img, save_hist

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import models.Net as Net
from dataset import shape_3d
from utils.utils import gradient

class model():
    def __init__(self, sdf_model, ray_tracer):
        self.ray_tracer = ray_tracer
        self.sdf_model = sdf_model
        self.sdf_model.eval()

    def render(self, uv, pose, intrinsics, object_mask, scale):
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        sdf = lambda x: self.sdf_model.decoder(x/scale)[:, 0] * scale
        points, network_object_mask, dists, iters = self.ray_tracer(sdf=sdf, cam_loc=cam_loc,
                                                                    object_mask=object_mask,ray_directions=ray_dirs)
        return dists, network_object_mask, iters, points
    

def get_sdf_model(args, trained_model_path, dataset_file_path, device="cuda"):
    train_set = shape_3d.ReconDataset(
        file_path = dataset_file_path,
        n_points=args.n_points,
        n_samples=args.n_iterations,
        grid_res=args.grid_res,
        grid_range=args.grid_range,
        sample_type=args.nonmnfld_sample_type,
        sampling_std=args.nonmnfld_sample_std,
        n_random_samples=args.n_random_samples,
        resample=True,
        compute_sal_dist_gt=(
            True if "sal" in args.loss_type and args.loss_weights[5] > 0 else False
        ),
        scale_method=args.pcd_scale_method,
    )
    _, scale = train_set.get_cp_and_scale(scale_method=args.pcd_scale_method)

    model = Net.Network(
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

    model.to(device)
    if args.parallel:
        if device.type == "cuda":
            model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(trained_model_path, map_location=device, weights_only=True))
    return model, scale


def eval(conf, args, img_size, compute_iters=True, compute_depth=False, compute_grads=False):
    # get trained sdf & ray_tracer model
    sdf_model, scale = get_sdf_model(args, args.model_path, args.dataset_file_path)
    ray_tracer = RayTracing(**conf.get_config('model.ray_tracer'))
    # combined model
    m = model(sdf_model, ray_tracer)
    # generate camera pose and params for rendering
    poses, uv_batches, intrinsics = generate_eval_cams(img_size=img_size, radius=1.0, num_views=36, height=0.5, fov=60, batch_size=10000)
    object_mask = torch.ones(10000, device="cuda:0", dtype=torch.bool)
    width = img_size[0]

    for itr, pose in enumerate(poses):
        all_dists, all_masks, all_iters, all_grads = [], [], [], []
        for uv in uv_batches:
            with torch.no_grad():
                dists, masks, iters, points = m.render(uv, pose, intrinsics, object_mask, scale)
                all_masks.extend(masks.detach().cpu().numpy())
                all_dists.extend(dists.detach().cpu().numpy())
                all_iters.extend(iters.detach().cpu().numpy())
            if compute_grads:
                out = sdf_model.decoder(points.requires_grad_(True)/scale)[:, 0] * scale
                grad = torch.norm(gradient(points, out), dim=1)
                all_grads.extend(grad.detach().cpu().numpy())

        mask_map = np.array(all_masks).reshape(-1, width)
        if compute_iters:
            iters_map = np.array(all_iters).reshape(-1, width)
            iters_masked = np.where(mask_map == 0, 0, iters_map).flatten()
        if compute_grads:
            grad_map = np.array(all_grads).reshape(-1, width)
            grad_map_masked = np.where(mask_map == 0, np.nan, grad_map)
        if compute_depth:
            depth_map= np.array(all_dists).reshape(-1, width) 
            depth_map_masked = np.where(mask_map == 0, np.nan, depth_map)

        # save iteration Map
        save_map_img(iters_map, itr, args.model_name, args.output_path)
        # save hist
        save_hist(all_iters, itr, args.model_name, args.output_path)
              

if __name__ == '__main__':
    conf = ConfigFactory.parse_file('ray_tracer.conf')
    img_size = [1600, 1200]
    args = parser.get_raytracer_args()
    eval(conf, args, img_size) 