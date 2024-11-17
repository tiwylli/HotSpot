import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

# Set Times New Roman as the font
rcParams["font.family"] = "Times New Roman"

# Set the directories containing the images
gt_dir = "raw_figures/2d_gt"
hotspot_dir = "raw_figures/2d_hotspot"
output_plot_path = "2d_corrected_plot_with_titles_aligned.png"

# Define the cropping box (x_start, y_start, x_end, y_end)
crop_box = (80, 100, 688, 719)


# Function to crop an image and return it as a NumPy array
def crop_image(image_path):
    with Image.open(image_path) as img:
        # Check if the image is 800x800
        if img.size == (800, 800):
            cropped_img = img.crop(crop_box)
            return cropped_img
        else:
            print(f"Skipped (not 800x800): {image_path}")
            return None


# List all image files in the directories
gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])
hotspot_files = sorted([f for f in os.listdir(hotspot_dir) if f.endswith(".png")])

# Ensure the two lists align; assume they're matched by index
assert len(gt_files) == len(
    hotspot_files
), "GT and Hotspot directories must have the same number of images."

# Create a 4x7 subplot
fig, axes = plt.subplots(
    4, 7, figsize=(12, 10), gridspec_kw={"hspace": -0.3, "wspace": 0.05}
)  # Reduce gaps between rows and columns

# Adjust vertical alignment for row labels
row_y_positions = [0.76, 0.58, 0.41, 0.23]  # Adjusted to match the rows properly
row_labels = ["GT", "Ours", "GT", "Ours"]

# Add row labels
for row, label in enumerate(row_labels):
    # Add text to the left side of each row
    fig.text(
        x=0.11,  # X position (left margin)
        y=row_y_positions[row],  # Y position adjusted for alignment
        s=label,  # Text to display
        va="center",  # Vertical alignment
        ha="center",  # Horizontal alignment
        fontsize=12,  # Font size
        rotation=90,  # Rotate vertically
    )

# Iterate through the GT and Hotspot images and plot them
for idx in range(len(gt_files)):
    col = idx % 7  # Column index
    row_gt = (idx // 7) * 2  # GT image row index
    row_hotspot = row_gt + 1  # Hotspot image row index

    # Crop and plot GT image
    gt_path = os.path.join(gt_dir, gt_files[idx])
    cropped_gt_img = crop_image(gt_path)
    if cropped_gt_img:
        ax_gt = axes[row_gt, col]
        ax_gt.imshow(cropped_gt_img)
        ax_gt.axis("off")

    # Crop and plot corresponding Hotspot image
    hotspot_path = os.path.join(hotspot_dir, hotspot_files[idx])
    cropped_hotspot_img = crop_image(hotspot_path)
    if cropped_hotspot_img:
        ax_hotspot = axes[row_hotspot, col]
        ax_hotspot.imshow(cropped_hotspot_img)
        ax_hotspot.axis("off")

    # Add a shared title for both GT and Hotspot images
    image_name = os.path.splitext(gt_files[idx])[0]
    if row_gt == 0 or row_gt == 2:  # Titles on both the first and third rows
        ax_gt.set_title(image_name, fontsize=12)

# Hide unused subplots (if any)
for ax in axes.flat:
    if not ax.has_data():
        ax.axis("off")

# Adjust layout and save the plot
plt.tight_layout(pad=0.5)  # Reduce padding for a tighter layout
plt.savefig(output_plot_path, dpi=300)
plt.show()
