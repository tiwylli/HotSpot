import os
import re
from PIL import Image
import matplotlib.pyplot as plt


def get_last_image_in_folder(folder_path):
    """Get the last sdf_ image in a folder, based on numerical sorting."""
    images = [
        img
        for img in os.listdir(folder_path)
        if img.startswith("sdf_") and img.endswith(".png")
    ]
    if not images:
        return None
    images.sort(
        key=lambda x: int(re.split(r"_|\.", x)[1])
    )  # Extracting number from sdf_XXXXX.png
    return os.path.join(folder_path, images[-1])


def visualize_last_images(base_path, shape_types, folder_name):
    """Visualize the last images for each shape type in a 2x7 grid and save the image."""
    fig, axs = plt.subplots(2, 7, figsize=(20, 10))
    fig.suptitle(f"Last Training Images for {folder_name}", fontsize=16)

    for idx, shape in enumerate(shape_types):
        row = idx // 7
        col = idx % 7
        vis_folder = os.path.join(base_path, folder_name, shape, "vis")

        last_image_path = get_last_image_in_folder(vis_folder)
        if last_image_path:
            img = Image.open(last_image_path)
            axs[row, col].imshow(img)
            axs[row, col].set_title(shape)
        else:
            axs[row, col].text(0.5, 0.5, "No Image", ha="center", va="center")

        axs[row, col].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Saving the figure
    output_folder = os.path.join(base_path, folder_name)
    output_path = os.path.join(output_folder, f"{folder_name}_combined_image.png")
    plt.savefig(output_path)
    print(f"Combined image saved at {output_path}")

    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python compose_sdf_vis.py <folder_name>")
        sys.exit(1)

    folder_name = sys.argv[1]
    base_path = "log/2D"  # Modify this if needed to point to the correct log folder
    shape_types = [
        "seaurchin",
        "L",
        "circle",
        "button",
        "target",
        "bearing",
        "snake",
        "peace",
        "boomerangs",
        "fragments",
        "house",
        "square",
        "snowflake",
        "starhex",
    ]

    visualize_last_images(base_path, shape_types, folder_name)
