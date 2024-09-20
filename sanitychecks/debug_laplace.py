import numpy as np
import plotly.graph_objects as go

# draw 100 samples from the Laplace distribution and plot them
n_samples = 10000
loc = 0
scale = 1
laplace_samples = np.random.laplace(loc, scale, n_samples)
angle_samples = np.random.uniform(0, 2 * np.pi, n_samples)
samples = np.stack(
    [laplace_samples * np.cos(angle_samples), laplace_samples * np.sin(angle_samples)], axis=-1
)

# plot histogram of the samples
fig = go.Figure()
fig.add_trace(go.Histogram(x=laplace_samples))
fig.update_layout(
    xaxis_title_text="x",
    yaxis_title_text="Number",
    title_text="Histogram of Laplace samples",
)

# print sum of probabilities
print("sum of probabilities:", np.sum(laplace_samples))

# save the figure as png
fig.write_image("laplace_samples.png")
