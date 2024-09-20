import os, sys

import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import basic_shape_dataset2d
import plotly
import numpy as np

n_points = 1000
n_samples = 1
grid_res = 128
non_mnfld_sample_type = "gaussian"
shape_type = "circle"
show_ax = False
title_text = "Non-manifold points"

shape = basic_shape_dataset2d.get2D_dataset(
    n_points, n_samples, grid_res, non_mnfld_sample_type, 0.005, shape_type=shape_type
)

# mnfld_points = shape.get_mnfld_points()  # shape: (n_samples, n_points, 2)
# print("mnfld_points.shape:", mnfld_points.shape)
nonmnfld_points, nonmnfld_pdfs = shape.get_nonmnfld_points_and_pdfs()
print("nonmnfld_points.shape:", nonmnfld_points.shape)
print("nonmnfld_pdfs.shape:", nonmnfld_pdfs.shape)

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=nonmnfld_points[:, 0],
            y=nonmnfld_points[:, 1],
            z=nonmnfld_pdfs[:, 0],
            mode="markers",
            marker=dict(
                size=2,
                color=nonmnfld_pdfs[:, 0],  # Use z values for color
                colorscale="Viridis",  # Choose a colorscale
                colorbar=dict(title="Z value"),  # Add a colorbar
            ),
        ),
    ]
)
fig.show()
# save as html
# plotly.offline.plot(fig, filename="nonmnfld_points.html")

