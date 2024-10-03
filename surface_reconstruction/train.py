import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import shape_2d
import recon_dataset as dataset
import torch
import numpy as np
import models.Net as model
import models.Heat as heatModel
from models.losses import Loss
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import utils.utils as utils
import surface_reconstruction.parser as parser
import utils.visualizations as vis
from PIL import Image

args = parser.get_train_args()

file_path = os.path.join(args.data_dir, args.file_name)
log_dir = os.path.join(args.log_dir, args.file_name.split(".")[0])

# set up logging
log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(log_dir, args)
os.system("cp %s %s" % (__file__, log_dir))  # backup the current training file
os.system("cp %s %s" % ("./models/Net.py", log_dir))  # backup the models files
os.system("cp %s %s" % ("./models/losses.py", log_dir))  # backup the losses files

# Set up dataloader
torch.manual_seed(0)  # change random seed for training set (so it will be different from test set
np.random.seed(0)
if args.task == "3d":
    train_set = dataset.ReconDataset(
        file_path=file_path,
        n_points=args.n_points,
        n_samples=args.n_iterations,
        grid_res=args.grid_res,
        grid_range=args.grid_range,
        sample_type=args.nonmnfld_sample_type,
        sampling_std2=args.nonmnfld_sample_std2,
        n_random_samples=args.n_random_samples,
    )
    in_dim = 3
elif args.task == "2d":
    train_set = shape_2d.get2D_dataset(
        n_points=args.n_points,
        n_samples=args.n_iterations,
        grid_res=args.grid_res,
        grid_range=args.grid_range,
        sample_type=args.nonmnfld_sample_type,
        sampling_std=args.nonmnfld_sample_std2,
        n_random_samples=args.n_random_samples,
        resample=True,
        shape_type=args.shape_type,
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
device = torch.device("cuda")

# model = model.Network(
#     latent_size=args.latent_size,
#     in_dim=in_dim,
#     decoder_hidden_dim=args.decoder_hidden_dim,
#     nl=args.nl,
#     encoder_type=args.encoder_type,
#     decoder_n_hidden_layers=args.decoder_n_hidden_layers,
#     init_type=args.init_type,
#     neuron_type=args.neuron_type,
#     sphere_init_params=args.sphere_init_params,
# )
# Uncomment to use small model
model = heatModel.Net(radius_init=args.sphere_init_params[1])
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
    heat_decay=args.heat_decay,
    heat_lambda_decay=args.heat_lambda_decay,
)
num_batches = len(train_dataloader)

# Iteratively train the model
for batch_idx, data in enumerate(train_dataloader):
    model.zero_grad()
    model.train()

    mnfld_points, mnfld_n_gt, nonmnfld_points, nonmnfld_pdfs = (
        data["mnfld_points"].to(device),
        data["mnfld_normals"].to(device),
        data["nonmnfld_points"].to(device),
        data["nonmnfld_pdfs"].to(device),
    )

    mnfld_points.requires_grad_()
    nonmnfld_points.requires_grad_()

    output_pred = model(nonmnfld_points, mnfld_points)

    loss_dict, _ = criterion(output_pred, mnfld_points, nonmnfld_points, nonmnfld_pdfs, mnfld_n_gt)
    lr = torch.tensor(optimizer.param_groups[0]["lr"])
    loss_dict["lr"] = lr
    utils.log_losses(log_writer_train, batch_idx, num_batches, loss_dict)

    loss_dict["loss"].backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

    optimizer.step()

    # Log training stats and svae model
    if batch_idx % args.log_interval == 0:
        weights = criterion.weights
        utils.log_string(f"Current heat lambda: {criterion.heat_lambda}", log_file)
        utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
        utils.log_string(
            "Iteration: {:4d}/{} ({:.0f}%) Loss: {:.5f} = L_Mnfld: {:.5f} + "
            "L_NonMnfld: {:.5f} + L_Nrml: {:.5f} + L_Eknl: {:.5f} + L_Div: {:.5f} + L_Heat: {:.5f}".format(
                batch_idx,
                len(train_set),
                100.0 * batch_idx / len(train_dataloader),
                loss_dict["loss"].item(),
                weights[0] * loss_dict["sdf_term"].item(),
                weights[1] * loss_dict["inter_term"].item(),
                weights[2] * loss_dict["normal_term"].item(),
                weights[3] * loss_dict["eikonal_term"].item(),
                weights[4] * loss_dict["div_term"].item(),
                weights[6] * loss_dict["heat_term"].item(),
            ),
            log_file,
        )
        utils.log_string(
            "Iteration: {:4d}/{} ({:.0f}%) Unweighted L_s : L_Mnfld: {:.5f},  "
            "L_NonMnfld: {:.5f},  L_Nrml: {:.5f},  L_Eknl: {:.5f},  L_Div: {:.5f},  L_Heat: {:.5f}".format(
                batch_idx,
                len(train_set),
                100.0 * batch_idx / len(train_dataloader),
                loss_dict["sdf_term"].item(),
                loss_dict["inter_term"].item(),
                loss_dict["normal_term"].item(),
                loss_dict["eikonal_term"].item(),
                loss_dict["div_term"].item(),
                loss_dict["heat_term"].item(),
            ),
            log_file,
        )
        # utils.log_string(
        #     "Iteration: {:4d}/{} ({:.0f}%) Mean unweighted L_s : L_Mnfld: {:.5f},  "
        #     "L_NonMnfld: {:.5f},  L_Nrml: {:.5f},  L_Eknl: {:.5f},  L_Div: {:.5f},  L_Heat: {:.5f}".format(
        #         batch_idx,
        #         len(train_set),
        #         100.0 * batch_idx / len(train_dataloader),
        #         loss_dict["sdf_term"].item(),
        #         loss_dict["inter_term"].item(),
        #         loss_dict["normal_term"].item(),
        #         loss_dict["eikonal_term"].item() / (2 * args.grid_range) ** in_dim,
        #         loss_dict["div_term"].item(),
        #         loss_dict["heat_term"].item() / (2 * args.grid_range) ** in_dim,
        #     ),
        #     log_file,
        # )
        # Save model
        utils.log_string(f"saving model to file model_{batch_idx}.pth", log_file)
        torch.save(model.state_dict(), os.path.join(model_outdir, f"model_{batch_idx}.pth"))
        utils.log_string("", log_file)

    # Visualize SDF
    if batch_idx % args.vis_interval == 0:
        utils.log_string(f"Visualizing epoch {batch_idx}", log_file)
        x, y = np.linspace(
            -args.vis_grid_range, args.vis_grid_range, args.vis_grid_res
        ), np.linspace(-args.vis_grid_range, args.vis_grid_range, args.vis_grid_res)
        meshgrid = np.meshgrid(x, y)
        meshgrid = np.stack(meshgrid, axis=-1)
        if in_dim == 3:
            z = np.zeros((args.vis_grid_res, args.vis_grid_res, 1))
            meshgrid = np.concatenate([meshgrid, z], axis=-1)
        grid_points = torch.tensor(meshgrid, dtype=torch.float32).to(device)
        grid_points.requires_grad_(True)

        output_dir = os.path.join(log_dir, "vis")
        os.makedirs(output_dir, exist_ok=True)

        output_pred = model(grid_points)
        mnfld_points_pred = output_pred["nonmanifold_pnts_pred"]

        sdf_contour_img = vis.plot_contours(
            x=x,
            y=y,
            z=mnfld_points_pred.detach()
            .cpu()
            .numpy()
            .reshape(args.vis_grid_res, args.vis_grid_res),
            colorscale="Geyser",
            show_scale=True,
            show_ax=True,
            title_text=f"SDF, epoch {batch_idx}",
            grid_range=args.vis_grid_range,
            contour_interval=args.vis_contour_interval,
        )
        # Save the generated images
        img = Image.fromarray(sdf_contour_img)
        img.save(os.path.join(output_dir, "sdf_" + str(batch_idx).zfill(6) + ".png"))
        utils.log_string("", log_file)

    if "div" in args.loss_type:
        criterion.update_div_weight(batch_idx, args.n_iterations, args.div_decay_params)
    if args.heat_lambda_decay is not None:
        criterion.update_heat_lambda(batch_idx, args.n_iterations, args.heat_lambda_decay_params)
    if args.heat_decay is not None:
        criterion.update_heat_weight(batch_idx, args.n_iterations, args.heat_decay_params)
    
    scheduler.step()

# Save final model
utils.log_string("saving model to file model.pth", log_file)
torch.save(model.state_dict(), os.path.join(model_outdir, "model.pth"))

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
        grid_res=args.grid_res,
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
