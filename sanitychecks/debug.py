import os, sys

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import basic_shape_dataset2d
import plotly
import numpy as np
import torch
from collections import defaultdict
from models.Heat import Net
from models import Net

n_points = 64
n_samples = 5
grid_res = 8
non_mnfld_sample_type = "grid"
shape_type = "starAndHexagon"
show_ax = False
title_text = "Points"
std = 0.09
n_random_samples = 64
batch_size = 32

train_set = basic_shape_dataset2d.get2D_dataset(
    n_points,
    n_samples,
    grid_res,
    non_mnfld_sample_type,
    std,
    n_random_samples,
    batch_size,
    shape_type=shape_type,
)
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=True
)

print("train_set.n_samples:", train_set.n_samples)
print("train_set.n_points:", train_set.n_points)
print("train_set.points.shape:", train_set.points.shape)
print("train_set.mnfld_n.shape:", train_set.mnfld_n.shape)
print("train_set.nonmnfld_dist.shape:", train_set.nonmnfld_dist.shape)
print("train_set.nonmnfld_n.shape:", train_set.nonmnfld_n.shape)
print("train_set.nonmnfld_points.shape:", train_set.nonmnfld_points.shape)
print("train_set.nonmnfld_dist.shape:", train_set.nonmnfld_dist.shape)
print()

# for data in train_dataloader:
#     print(data["mnfld_points"].shape)
#     print(data["nonmnfld_points"].shape)
#     print()
# exit(0)

# nonmnfld_points, nonmnfld_pdfs = train_set.get_nonmnfld_points_and_pdfs()
# print("nonmnfld_points.shape:", nonmnfld_points.shape)
# print("nonmnfld_pdfs.shape:", nonmnfld_pdfs.shape)

latent_size = 0
decoder_hidden_dim = 128
nl = "softplus"
neuron_type = "linear"
decoder_n_hidden_layers = 4
init_type = "mfgi"

# =========
# Test Nets
# =========

# net = ImplicitNet()
# SINR = Net.Network(
#     latent_size=latent_size,
#     in_dim=2,
#     decoder_hidden_dim=decoder_hidden_dim,
#     nl=nl,
#     encoder_type="none",
#     neuron_type=neuron_type,
#     decoder_n_hidden_layers=decoder_n_hidden_layers,
#     init_type=init_type,
# )

# mnfld_points = torch.randn(1, 20, 2)
# nonmnfld_points = torch.randn(1, 30, 2)

# out = SINR(mnfld_points, nonmnfld_points)

# print(out["manifold_pnts_pred"].shape)
# print(out["nonmanifold_pnts_pred"].shape)

# out = net(nonmnfld_points)

# print(out.shape)

# exit(0)

data = defaultdict(list)

for i, batch in enumerate(train_dataloader):
    print(f"batch['points'].shape: {batch['points'].shape}")
    print(f"batch['nonmnfld_points'].shape: {batch['nonmnfld_points'].shape}")
    print(f"batch['nonmnfld_pdfs'].shape: {batch['nonmnfld_pdfs'].shape}")
    data["mnfld_points"].append(batch["mnfld_points"].squeeze(0).numpy())
    data["nonmnfld_points"].append(batch["nonmnfld_points"].squeeze(0).numpy())
    data["nonmnfld_pdfs"].append(batch["nonmnfld_pdfs"].squeeze(0).numpy())
    data["colors"].append(np.ones(batch["nonmnfld_points"].shape[1]) * i / len(train_dataloader))

print()

data["mnfld_points"] = np.concatenate(data["mnfld_points"], axis=0)
data["nonmnfld_points"] = np.concatenate(data["nonmnfld_points"], axis=0)
data["nonmnfld_pdfs"] = np.concatenate(data["nonmnfld_pdfs"], axis=0)
data["colors"] = np.concatenate(data["colors"], axis=0)

fig = go.Figure(
    go.Scatter(
        x=data["mnfld_points"][:, 0],
        y=data["mnfld_points"][:, 1],
        mode="markers",
        marker=dict(
            size=2,
            color=data["colors"],
            colorscale="Viridis",
        ),
    )
)

fig.update_layout(
    xaxis=dict(range=[-1, 1]),
    yaxis=dict(range=[-1, 1])
)

# fig = go.Figure(
#     go.Scatter3d(
#         x=data["mnfld_points"][:, 0],
#         y=data["mnfld_points"][:, 1],
#         # z=data["nonmnfld_pdfs"][:, 0],
#         # z=0,
#         mode="markers",
#         marker=dict(
#             size=2,
#             color=data["colors"],
#             colorscale="Viridis",
#         ),
#     )
# )
fig.show()

# save as html
# plotly.offline.plot(fig, filename="nonmnfld_points.html")
