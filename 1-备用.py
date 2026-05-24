import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent
import matplotlib.patches as patches

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points

# Global variable to store clicked points
clicked_points = []
point_labels = []

import matplotlib

matplotlib.use('TkAgg')  # Use the TkAgg backend

# Initialize lists to store patches and points
points = []
circles = []
click_index = 0  # Initialize a counter for the click index

def onclick(event: MouseEvent):
    global clicked_points, points, circles, radius, click_index
    if event.inaxes is not None:  # Check if the click is inside the axes
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            click_index += 1  # Increment the click index
            clicked_points.append((x, y))
            print(f"Clicked at index {click_index}: ({x}, {y})")  # Print click index and coordinates

            # Add a red dot at the click position
            point, = ax.plot(x, y, 'ro')  # Store the plot element for later reference
            points.append(point)

            # Add a circle around the clicked point to show the range
            circle = patches.Circle((x, y), radius=radius, edgecolor='blue', facecolor='none', linestyle='--', linewidth=1)
            ax.add_patch(circle)
            circles.append(circle)

            # Add a text label next to the point indicating its index
            ax.text(x, y, f'{click_index}', color='red', fontsize=16, ha='right', va='bottom')

            plt.draw()  # Redraw the plot to show the new point, circle, and text

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=False,
        default="D:/medical_paper/Inpaint-Anything/example/2011-1/breast/201112260191.jpg",
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=False,
        default="click", choices=["click", "key_in"],
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=False,
        # default=[200, 450],
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='*', required=False,
        default=[],
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, required=False,
        default=7,
        help="Dilate kernel size. Default: 17",
    )
    parser.add_argument(
        "--output_dir", type=str, required=False,
        default="D:/medical_paper/Inpaint-Anything/results/2011-1/breast",
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h'",
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=False,
        default="./pretrained_models/sam_vit_h_4b8939.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        default="D:/medical_paper/Inpaint-Anything/pretrained_models/big-lama", required=False,
        help="The path to the lama checkpoint.",
    )

if __name__ == "__main__":
    """Example usage:
    python remove_anything.py
        --input_img FA_demo/FA1_dog.png
        --coords_type click
        --dilate_kernel_size 15
        --output_dir ./results
        --sam_model_type "vit_h"
        --sam_ckpt sam_vit_h_4b8939.pth
        --lama_config lama/configs/prediction/default.yaml
        --lama_ckpt big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set the radius based on dilate_kernel_size
    radius = args.dilate_kernel_size

    if args.coords_type == "click":
        # Load the image and display it for interactive clicking
        img = load_img_to_array(args.input_img)
        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.axis('off')

        # Connect the click event to the onclick function
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()  # Show the interactive plot

        # After clicking, ask for labels in one go
        if clicked_points:
            latest_coords = np.array(clicked_points)
            # Ask for labels
            for i in range(len(clicked_points)):
                while True:
                    try:
                        label = int(input(f"Enter label for point {i + 1} (1 or 0): "))
                        if label in [0, 1]:
                            point_labels.append(label)
                            break
                        else:
                            print("Invalid input. Label must be 0 or 1.")
                    except ValueError:
                        print("Invalid input. Please enter an integer.")
        else:
            raise ValueError("No points were clicked. Exiting...")

    elif args.coords_type == "key_in":
        latest_coords = np.array([args.point_coords])
        point_labels = args.point_labels

    # Ensure coordinates are in the correct format for the model
    img = load_img_to_array(args.input_img)
    masks, _, _ = predict_masks_with_sam(
        img,
        latest_coords,
        point_labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255

    # Dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # Visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # Path to the results
        mask_p = out_dir / f"mask_{idx}.jpg"
        img_points_p = out_dir / f"with_points.jpg"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # Save the mask
        save_array_to_img(mask, mask_p)

        # Save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), latest_coords, point_labels,
                    size=(width * 0.04) ** 2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Inpaint the masked image
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.jpg"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)
