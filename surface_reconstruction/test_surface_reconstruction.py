# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recon_dataset as dataset
import torch
import utils.visualizations as vis
import numpy as np
import models.Net as Net
import torch.nn.parallel
import utils.utils as utils
import surface_recon_args
from PIL import Image

# get training parameters
args = surface_recon_args.get_test_args()

file_path = os.path.join(args.dataset_path, args.file_name)
if args.export_mesh:
    outdir = os.path.join(os.path.dirname(args.logdir), "result_meshes")
    os.makedirs(outdir, exist_ok=True)
    output_ply_filepath = os.path.join(outdir, args.file_name)
# get data loader
torch.manual_seed(args.seed)
np.random.seed(args.seed)
test_set = dataset.ReconDataset(
    file_path,
    args.n_points,
    n_samples=1,
    res=args.grid_res,
    sample_type="grid",
    requires_dist=False,
)
test_dataloader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

# get model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
device = torch.device("cuda")

Net = Net.Network(
    latent_size=args.latent_size,
    in_dim=3,
    decoder_hidden_dim=args.decoder_hidden_dim,
    nl=args.nl,
    encoder_type=args.encoder_type,
    decoder_n_hidden_layers=args.decoder_n_hidden_layers,
    init_type=args.init_type,
    neuron_type=args.neuron_type,
)
if args.parallel:
    if device.type == "cuda":
        Net = torch.nn.DataParallel(Net)

model_dir = os.path.join(args.logdir, "trained_models")
trained_model_filename = os.path.join(model_dir, "model.pth")
Net.load_state_dict(torch.load(trained_model_filename, map_location=device, weights_only=True))
Net.to(device)
latent = None

print("Visualizing SDF value on a plane")
# generate 3d grid points
x, y = np.linspace(-args.grid_range, args.grid_range, args.grid_res), np.linspace(-args.grid_range, args.grid_range, args.grid_res)
meshgrid = np.meshgrid(x, y)
meshgrid = np.stack(meshgrid, axis=-1)
z = np.zeros((args.grid_res, args.grid_res, 1))
meshgrid = np.concatenate([meshgrid, z], axis=-1)
grid_points = torch.tensor(meshgrid, dtype=torch.float32).to(device)
print("grid_points.shape: ", grid_points.shape)
grid_points.requires_grad_(True)

output_dir = os.path.join(args.logdir, "vis")
os.makedirs(output_dir, exist_ok=True)

for epoch in args.epoch_n:
    model_filename = os.path.join(model_dir, "model_%d.pth" % (epoch))
    Net.load_state_dict(torch.load(model_filename, map_location=device, weights_only=True))
    Net.to(device)

    print("Converting implicit to level set epoch {}".format(epoch))

    output_pred = Net(grid_points)
    mnfld_points_pred = output_pred["nonmanifold_pnts_pred"]

    sdf_contour_img = vis.plot_contours(
        x=x,
        y=y,
        z=mnfld_points_pred.detach().cpu().numpy().reshape(args.grid_res, args.grid_res),
        colorscale="Geyser",
        show_scale=True,
        show_ax=True,
        title_text=f"SDF, epoch {epoch}",
    )
    # save the generated images
    img = Image.fromarray(sdf_contour_img)
    img.save(os.path.join(output_dir, "sdf_" + str(epoch).zfill(6) + ".png"))

print("Converting implicit to mesh for file {}".format(args.file_name))
cp, scale, bbox = test_set.cp, test_set.scale, test_set.bbox
test_set, test_dataloader, clean_points_gt, normals_gt, nonmnfld_points, data = (
    None,
    None,
    None,
    None,
    None,
    None,
)  # free up memory
mesh_dict = utils.implicit2mesh(
    Net.decoder,
    latent,
    args.grid_res,
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
