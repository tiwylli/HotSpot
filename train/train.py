import os
import sys

from PIL import Image

import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import shape_2d, shape_3d
import models.Net as Net
import models.SkipNet as SkipNet
from models.losses import Loss

import utils.utils as utils
import utils.visualizations as vis
from utils import parser
import copy


def occupancy_to_sdf(occupancy, epsilon):
    sdf = -(epsilon**0.5) * torch.log(1 - torch.abs(occupancy)) * torch.sign(occupancy)
    return sdf


def unpack_pred_dists(pred, args):
    mnfld_pred = pred["manifold_pnts_pred"]
    nonmnfld_pred = pred["nonmanifold_pnts_pred"]
    if args.loss_type == "phase":
        mnfld_pred = occupancy_to_sdf(mnfld_pred, args.phase_epsilon)
        nonmnfld_pred = occupancy_to_sdf(nonmnfld_pred, args.phase_epsilon)

    return mnfld_pred, nonmnfld_pred


def visualize_model(
    x_grid,
    y_grid,
    mnfld_points,
    vis_grid_pred,
    mnfld_points_pred,
    vis_grid_dists_gt,
    data,
    batch_idx,
    args,
    shape=None,
):
    if args.vis_normals:
        mnfld_normals_pred = utils.gradient(mnfld_points, mnfld_points_pred)
        mnfld_normals_pred = mnfld_normals_pred / torch.norm(
            mnfld_normals_pred, dim=-1, keepdim=True
        )
        mnfld_normals_gt = data["mnfld_normals_gt"]

    batch_idx_suffix = "final" if batch_idx == "final" else str(batch_idx).zfill(6)

    sdf_contour_img = vis.plot_contours(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=vis_grid_pred.detach().cpu().numpy().reshape(args.vis_grid_res, args.vis_grid_res),
        mnfld_points=(
            mnfld_points[0][: args.n_vis_normals].detach().cpu().numpy()
            if args.vis_normals
            else None
        ),
        mnfld_normals=(
            mnfld_normals_pred[0][: args.n_vis_normals].detach().cpu().numpy()
            if args.vis_normals
            else None
        ),
        mnfld_normals_gt=mnfld_normals_gt[0][: args.n_vis_normals] if args.vis_normals else None,
        colorscale="RdBu_r",
        show_scale=True,
        show_ax=True,
        title_text=f"SDF, epoch {batch_idx}",
        grid_range=args.vis_grid_range,
        contour_interval=args.vis_contour_interval,
        contour_range=args.vis_contour_range,
        gt_traces=(
            shape.get_trace(color="rgb(128, 128, 128)")
            if args.vis_gt_shape and shape is not None
            else []
        ),
    )
    img = Image.fromarray(sdf_contour_img)
    img.save(os.path.join(output_dir, "sdf_" + batch_idx_suffix + ".png"))

    if args.vis_heat:
        vis_grid_heat = np.exp(
            -args.heat_lambda
            * np.abs(
                vis_grid_pred.detach().cpu().numpy().reshape(args.vis_grid_res, args.vis_grid_res)
            )
        )
        heat_contour_img = vis.plot_contours(
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=vis_grid_heat,
            mnfld_points=None,
            mnfld_normals=None,
            mnfld_normals_gt=None,
            colorscale="Peach",
            show_scale=True,
            show_ax=True,
            title_text=f"Heat, epoch {batch_idx}",
            grid_range=args.vis_grid_range,
            contour_interval=args.vis_contour_interval,
            contour_range=[0, 1],
        )
        img = Image.fromarray(heat_contour_img)
        img.save(os.path.join(output_dir, "heat_" + batch_idx_suffix + ".png"))

    if args.vis_diff:
        vis_grid_diff = (
            vis_grid_pred.squeeze().detach().cpu().numpy() - vis_grid_dists_gt.squeeze()
        ).reshape(args.vis_grid_res, args.vis_grid_res)
        diff_contour_img = vis.plot_contours(
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=vis_grid_diff,
            mnfld_points=None,
            mnfld_normals=None,
            mnfld_normals_gt=None,
            colorscale="Tropic",
            show_scale=True,
            show_ax=True,
            title_text=f"Diff, epoch {batch_idx}",
            grid_range=args.vis_grid_range,
            contour_interval=args.vis_contour_interval,
        )
        img = Image.fromarray(diff_contour_img)
        img.save(os.path.join(output_dir, "diff_" + batch_idx_suffix + ".png"))

    if args.vis_final:
        gt_contour_img = vis.plot_contours(
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=vis_grid_dists_gt.reshape(args.vis_grid_res, args.vis_grid_res),
            mnfld_points=None,
            mnfld_normals=None,
            mnfld_normals_gt=None,
            colorscale="RdBu_r",
            show_scale=True,
            show_ax=True,
            title_text=f"GT",
            grid_range=args.vis_grid_range,
            contour_interval=args.vis_contour_interval,
            contour_range=args.vis_contour_range,
        )
        img = Image.fromarray(gt_contour_img)
        img.save(os.path.join(output_dir, "gt.png"))

    utils.log_string("", log_file)


if __name__ == "__main__":
    args = parser.get_train_args()

    if not args.train and not args.eval:
        raise ValueError("Please specify either --train or --eval, or both.")

    file_path = os.path.join(args.data_dir, args.file_name)
    log_dir = os.path.join(
        args.log_dir, args.file_name.split(".")[0]
    )  # Concatenate the log directory with the file name if file name is given

    # set up logging
    log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(log_dir, args)
    os.system("cp %s %s" % (__file__, log_dir))  # backup the current training file
    os.system("cp %s %s" % ("./models/Net.py", log_dir))  # backup the models files
    os.system("cp %s %s" % ("./models/losses.py", log_dir))  # backup the losses files

    # Set up dataloader
    torch.manual_seed(0)
    # change random seed for training set (so it will be different from test set
    np.random.seed(0)
    if args.task == "3d":
        train_set = shape_3d.ReconDataset(
            file_path=file_path,
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
        in_dim = 3
    elif args.task == "2d":
        train_set = shape_2d.get2D_dataset(
            shape_type=args.shape_type,
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
        )
        in_dim = 2

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.loss_type == "phase":
        model = SkipNet.SkipNet(
            in_dim=in_dim,
            nl=args.nl,
            ff_layers=[],
            clamp=args.skipnet_clamp,
        )
    else:
        model = Net.Network(
            latent_size=args.latent_size,
            in_dim=in_dim,
            decoder_hidden_dim=args.decoder_hidden_dim,
            nl=args.nl,
            encoder_type=args.encoder_type,
            decoder_n_hidden_layers=args.decoder_n_hidden_layers,
            init_type=args.init_type,
            neuron_type=args.neuron_type,
            sphere_init_params=args.sphere_init_params,
            n_repeat_period=args.n_repeat_period,
        )

    # Uncomment to use small model
    # model = heatModel.Net(radius_init=args.sphere_init_params[1])
    model.to(device)
    if args.parallel:
        if device.type == "cuda":
            model = torch.nn.DataParallel(model)
    n_parameters = utils.count_parameters(model)
    utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)

    # Set up optimizer, scheduler, and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1.0)  # Does nothing

    criterion = Loss(
        weights=args.loss_weights,
        loss_type=args.loss_type,
        div_decay=args.div_decay,
        div_type=args.div_type,
        heat_lambda=args.heat_lambda,
        phase_epsilon=args.phase_epsilon,
        heat_decay=args.heat_decay,
        eikonal_decay=args.eikonal_decay,
        heat_lambda_decay=args.heat_lambda_decay,
        boundary_coef_decay=args.boundary_coef_decay,
        importance_sampling=args.importance_sampling,
    )
    num_batches = len(train_dataloader)

    scale = 1.0
    default_scale = 1.0
    if in_dim == 3:
        cp, scale = train_set.cp, train_set.scale
        default_cp, default_scale = train_set.get_cp_and_scale(scale_method="default")

    # Set up visualization grid
    x_vis, y_vis = np.linspace(
        -args.vis_grid_range, args.vis_grid_range, args.vis_grid_res
    ), np.linspace(-args.vis_grid_range, args.vis_grid_range, args.vis_grid_res)
    x, y = np.linspace(-args.vis_grid_range, args.vis_grid_range, args.vis_grid_res), np.linspace(
        -args.vis_grid_range, args.vis_grid_range, args.vis_grid_res
    )
    x, y = x * default_scale / scale, y * default_scale / scale
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()
    vis_grid_points = np.stack([xx, yy], axis=-1)
    if in_dim == 3:
        z = np.zeros((args.vis_grid_res**2, 1))
        vis_grid_points = np.concatenate([vis_grid_points, z], axis=-1)
    vis_grid_points = vis_grid_points[None, ...]  # (1, grid_res ** 2, dim)
    vis_grid_points = torch.tensor(vis_grid_points, dtype=torch.float32).to(device)
    vis_grid_points.requires_grad_()

    if not args.vis_final:
        # Iteratively train the model
        for batch_idx, data in enumerate(train_dataloader):
            # Load data
            mnfld_points = data["mnfld_points"].to(device)
            mnfld_normals_gt = data["mnfld_normals_gt"].to(device)
            nonmnfld_points = data["nonmnfld_points"].to(device)
            nonmnfld_pdfs = data["nonmnfld_pdfs"].to(device)
            # nonmnfld_dists_gt = data["nonmnfld_dists_gt"].to(device)
            # grid_dists_gt = data["grid_dists_gt"].to(device)

            # Conditionally load nonmnfld_dists_sal if it exists in the data
            nonmnfld_dists_sal = data.get("nonmnfld_dists_sal", None)
            if nonmnfld_dists_sal is not None:
                nonmnfld_dists_sal = nonmnfld_dists_sal.to(device)

            mnfld_points.requires_grad_()
            nonmnfld_points.requires_grad_()
            # Save model before updating weights
            if args.train:
                if batch_idx % args.log_interval == 0:
                    utils.log_string(f"saving model to file model_{batch_idx}.pth", log_file)
                    torch.save(
                        model.state_dict(), os.path.join(model_outdir, f"model_{batch_idx}.pth")
                    )

            # Visualize SDF
            if not args.vis_final and args.eval and batch_idx % args.vis_interval == 0:
                if not args.train:
                    model_path = os.path.join(log_dir, "trained_models", f"model_{batch_idx}.pth")
                    model.load_state_dict(torch.load(model_path, weights_only=True))

                utils.log_string(f"Visualizing epoch {batch_idx}", log_file)
                output_dir = os.path.join(log_dir, "vis")
                os.makedirs(output_dir, exist_ok=True)
                model_copy = copy.deepcopy(model)
                model_copy.eval()
                vis_pred = model_copy(vis_grid_points, mnfld_points)
                vis_grid_dists_gt, _ = train_set.get_points_distances_and_normals(
                    vis_grid_points[0].detach().cpu().numpy()
                )  # (vis_grid_res * vis_grid_res, 1)
                if vis_grid_dists_gt is not None:
                    vis_grid_dists_gt = vis_grid_dists_gt.reshape(
                        args.vis_grid_res, args.vis_grid_res
                    )

                mnfld_points_pred, vis_grid_pred = unpack_pred_dists(vis_pred, args)

                visualize_model(
                    x_grid=x_vis,
                    y_grid=y_vis,
                    mnfld_points=mnfld_points,
                    vis_grid_pred=vis_grid_pred * scale / default_scale,
                    mnfld_points_pred=mnfld_points_pred,
                    vis_grid_dists_gt=vis_grid_dists_gt,
                    data=data,
                    batch_idx=batch_idx,
                    args=args,
                    shape=train_set,
                )

            if args.train:
                # reset grad of mnfld_points and nonmnfld_points
                model.zero_grad()
                model.train()

                # Compute losses on samples
                output_pred = model(nonmnfld_points, mnfld_points)
                loss_dict, _ = criterion(
                    output_pred,
                    mnfld_points,
                    nonmnfld_points,
                    nonmnfld_pdfs,
                    mnfld_normals_gt,
                    None,
                    nonmnfld_dists_sal,
                )
                # Updatae learning rate
                lr = torch.tensor(optimizer.param_groups[0]["lr"])
                loss_dict["lr"] = lr
                # Log losses on samples
                utils.log_losses(log_writer_train, batch_idx, num_batches, loss_dict)

                # Backpropagate and update weights
                loss_dict["loss"].backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

                # Log training stats and save model
                if batch_idx % args.log_interval == 0:
                    weights = criterion.weights
                    utils.log_string(f"Current heat lambda: {criterion.heat_lambda}", log_file)
                    utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
                    # Log weighted losses
                    utils.log_string(
                        "Iteration: {:4d}/{} ({:.0f}%) Loss: {:.5f} = L_Mnfld: {:.5f} + "
                        "L_NonMnfld: {:.5f} + L_Nrml: {:.5f} + L_Eknl: {:.5f} + L_Div: {:.5f} + L_SAL: {:.5f} + L_Heat: {:.5f}".format(
                            batch_idx,
                            len(train_set),
                            100.0 * batch_idx / len(train_dataloader),
                            loss_dict["loss"].item(),
                            weights[0] * loss_dict["boundary_term"].item(),
                            weights[1] * loss_dict["inter_term"].item(),
                            weights[2] * loss_dict["normal_term"].item(),
                            weights[3] * loss_dict["eikonal_term"].item(),
                            weights[4] * loss_dict["div_term"].item(),
                            weights[5] * loss_dict["sal_term"].item(),  # add SAL term here
                            weights[6] * loss_dict["heat_term"].item(),
                        ),
                        log_file,
                    )
                    # Log unweighted losses
                    utils.log_string(
                        "Iteration: {:4d}/{} ({:.0f}%) Unweighted L_s : L_Mnfld: {:.5f},  "
                        "L_NonMnfld: {:.5f},  L_Nrml: {:.5f},  L_Eknl: {:.5f},  L_Div: {:.5f},  L_SAL: {:.5f}, L_Heat: {:.5f},  L_Diff: {:.5f}".format(
                            batch_idx,
                            len(train_set),
                            100.0 * batch_idx / len(train_dataloader),
                            loss_dict["boundary_term"].item(),
                            loss_dict["inter_term"].item(),
                            loss_dict["normal_term"].item(),
                            loss_dict["eikonal_term"].item(),
                            loss_dict["div_term"].item(),
                            loss_dict["sal_term"].item(),  # add SAL term here
                            loss_dict["heat_term"].item(),
                            loss_dict["diff_term"].item(),
                        ),
                        log_file,
                    )
                    utils.log_string("", log_file)

            # Update lambda
            if args.heat_lambda_decay is not None:
                criterion.update_heat_lambda(
                    batch_idx, args.n_iterations, args.heat_lambda_decay_params
                )

            # Update weights
            if args.train:
                if "div" in args.loss_type:
                    criterion.update_div_weight(batch_idx, args.n_iterations, args.div_decay_params)
                if args.heat_decay is not None:
                    criterion.update_heat_weight(
                        batch_idx, args.n_iterations, args.heat_decay_params
                    )
                if args.eikonal_decay is not None:
                    criterion.update_eikonal_weight(
                        batch_idx, args.n_iterations, args.eikonal_decay_params
                    )
                if args.boundary_coef_decay is not None:
                    criterion.update_boundary_coef(
                        batch_idx, args.n_iterations, args.boundary_coef_decay_params
                    )

                scheduler.step()

    # Save final model
    if args.train and not args.vis_final:
        utils.log_string("Saving final model to file model.pth", log_file)
        torch.save(model.state_dict(), os.path.join(model_outdir, "model.pth"))

    # Visualize final pth if exists
    if args.eval and args.vis_final:
        test_dataloader = torch.utils.data.DataLoader(
            train_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
        )
        test_data = next(iter(test_dataloader))
        mnfld_points = test_data["mnfld_points"].to(device)
        mnfld_points.requires_grad_()
        model_path = os.path.join(log_dir, "trained_models", f"model.pth")
        if torch.cuda.is_available():
            map_location = torch.device("cuda")
        else:
            map_location = torch.device("cpu")
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=map_location))

        utils.log_string(f"Visualizing final model", log_file)

        output_dir = os.path.join(log_dir, "vis")
        os.makedirs(output_dir, exist_ok=True)

        vis_pred = model(vis_grid_points, mnfld_points)
        vis_grid_dists_gt, _ = train_set.get_points_distances_and_normals(
            vis_grid_points[0].detach().cpu().numpy()
        )  # (vis_grid_res * vis_grid_res, 1)

        mnfld_points_pred, vis_grid_pred = unpack_pred_dists(vis_pred, args)

        visualize_model(
            x_grid=x_vis,
            y_grid=y_vis,
            mnfld_points=mnfld_points,
            vis_grid_pred=vis_grid_pred * scale / default_scale,
            mnfld_points_pred=mnfld_points_pred,
            vis_grid_dists_gt=vis_grid_dists_gt,
            data=test_data,
            batch_idx="final",
            args=args,
            shape=train_set,
        )

    # Save video
    if args.eval:
        if args.save_video and not args.vis_final:
            vis.save_video(output_dir, "sdf.mp4", "sdf_*.png")
            if args.vis_heat:
                vis.save_video(output_dir, "heat.mp4", "heat_*.png")

        # Convert implicit to mesh
        if in_dim == 3:
            print("Converting implicit to mesh for file {}".format(args.file_name))
            output_ply_filepath = os.path.join(log_dir, "output.ply")
            cp, scale, bbox = train_set.cp, train_set.scale, train_set.bbox
            test_set, test_dataloader, clean_points_gt, normals_gt, nonmnfld_points, data = (
                None,
                None,
                None,
                None,
                None,
                None,
            )  # free up memory
            mesh_dict = utils.implicit2mesh(
                decoder=model.decoder,
                latent=None,
                grid_res=256,
                translate=-cp,
                scale=1 / scale,
                get_mesh=True,
                device=device,
                bbox=bbox,
            )
            vis.plot_mesh(
                mesh_dict["mesh_trace"],
                mesh=mesh_dict["mesh_obj"],
                output_ply_path=output_ply_filepath,
                show_ax=False,
                title_txt=args.file_name.split(".")[0],
                show=False,
            )

            print("Conversion complete.")
