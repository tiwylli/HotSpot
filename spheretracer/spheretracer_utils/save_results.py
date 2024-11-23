import numpy as np
import os
import matplotlib.pyplot as plt

def save_map_img(map, itr, model_name, output_path, map_type):  
    if map_type == "iteration":
        cmap = 'plasma'
        label = 'iterations'
        colorbar=True
    elif map_type == "depth":
        cmap = 'grey'
        colorbar=False
    elif map_type == "normal":
        cmap = 'rainbow'
        colorbar=False
    
    plt.figure(figsize=(6, 6), dpi=100) 
    plt.imshow(map, cmap=cmap, interpolation='nearest')
    plt.tick_params(axis='both', which='both', length=0,  labelbottom=False, labelleft=False) 
    if colorbar:
        cbar = plt.colorbar(label=label) 
        cbar.set_label(label, fontsize=15)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"{model_name}_map_{map_type}_{itr}.png"))
    plt.close()

def save_hist(vals, itr, model_name, output_path):
    plt.hist(vals, bins=30, color='gray', edgecolor='black', density=True)
    plt.title(f'{model_name}: Iteration Distribution (pose {itr})', fontsize=15)
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.ylim(0, 0.7) 
    plt.tick_params(axis='both', which='major', labelsize=15)
    textstr = f'Mean: {np.mean(vals):.2f}\nVariance: {np.var(vals):.2f}'
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
            fontsize=25, verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8))
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"{model_name}_hist_{itr}.png"))
    plt.close()