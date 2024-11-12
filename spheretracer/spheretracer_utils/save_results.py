import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def save_map_img(iters_map, itr, model_name, output_path):
    #norm=LogNorm(vmin=1, vmax=30)
    plt.imshow(iters_map, cmap='plasma', vmin=1, vmax=30)
    plt.title(f'{model_name}: iteration map (pose {itr})')
    plt.colorbar(label='Steps')
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"{model_name}_map_{itr}.png"))
    plt.close()

def save_hist(vals, itr, model_name, output_path):
    plt.hist(vals, bins=30, color='gray', edgecolor='black', density=True)
    plt.title(f'{model_name}: Iteration Distribution (pose {itr})')
    plt.xlabel('Iterations')
    plt.ylim(0, 0.5) 
    textstr = f'Mean: {np.mean(vals):.2f}\nVariance: {np.var(vals):.2f}'
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8))
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"{model_name}_hist_{itr}.png"))
    plt.close()