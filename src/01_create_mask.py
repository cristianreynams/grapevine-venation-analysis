import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

# =====================================
# CONFIG
# =====================================

INPUT_ROOT = "data/processed/cropped"

# More restrictive than ROI crop
LOWER_GREEN = np.array([18, 38, 23])
UPPER_GREEN = np.array([110, 255, 255])

# Smaller kernel for finer edges
KERNEL_SIZE = 9

# =====================================
# CORE FUNCTION
# =====================================

def create_mask(image_path):

    # =====================================
    # LOAD IMAGE
    # =====================================

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not load {image_path}")

    # =====================================
    # BLUR
    # =====================================

    blur = cv2.bilateralFilter(
        img,
        d=7,
        sigmaColor=50,
        sigmaSpace=50
    )

    # =====================================
    # RGB -> HSV
    # =====================================

    hsv = cv2.cvtColor(
        blur,
        cv2.COLOR_BGR2HSV
    )

    # =====================================
    # HSV THRESHOLD
    # green + saturation
    # =====================================

    mask = cv2.inRange(
        hsv,
        LOWER_GREEN,
        UPPER_GREEN
    )

    # =====================================
    # MORPHOLOGY
    # =====================================

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5,5) # Smaller kernel
    )

    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5,5)
    )

    # Fill small internal gaps
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    # Remove small external noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open
    )

    # =====================================
    # LARGEST CONNECTED COMPONENT
    # =====================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 1:

        raise ValueError(
            f"No foreground detected: {image_path}"
        )

    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    final_mask = labels == largest_label

    final_mask = binary_fill_holes(
        final_mask
    )

    final_mask = np.uint8(final_mask) * 255

    
    return {
        "original": img,
        "mask": final_mask
    }

# =====================================
# SAVE FUNCTION
# =====================================

def save_mask(mask, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    cv2.imwrite(path, mask)

# =====================================
# DEBUG MODE
# =====================================

def debug_mode(folder_name):

    input_dir = os.path.join(
        INPUT_ROOT,
        folder_name
    )

    output_dir = os.path.join(
        "data/processed/masks",
        "debug_img"
    )

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff"
        ))
    ])[:4]

    results = []

    for file in image_files:

        image_path = os.path.join(
            input_dir,
            file
        )

        result = create_mask(image_path)

        # =====================================
        # SAVE MASK
        # =====================================

        save_path = os.path.join(
            output_dir,
            f"mask_{file}"
        )

        save_mask(
            result["mask"],
            save_path
        )
        
        original_rgb = cv2.cvtColor(
            result["original"],
            cv2.COLOR_BGR2RGB
        )

        results.append((
            file,
            original_rgb,
            result["mask"]
        ))

    
    # =====================================
    # VISUALIZATION
    # =====================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 6)
    )

    axes = axes.flatten()

    for i, (name, original, mask) in enumerate(results):

        # =====================================
        # ORIGINAL
        # =====================================

        axes[i * 2].imshow(original)

        axes[i * 2].set_title(
            f"{name}\nOriginal",
            fontsize=9
        )

        axes[i * 2].axis("off")

        # =====================================
        # MASK
        # =====================================

        axes[i * 2 + 1].imshow(
            mask,
            cmap="gray"
        )

        axes[i * 2 + 1].set_title(
            f"{name}\nMask",
            fontsize=9
        )

        axes[i * 2 + 1].axis("off")

    plt.tight_layout(
        pad=1.0,
        w_pad=0.5,
        h_pad=1.0
    )       

    plt.savefig(
        os.path.join(
            output_dir,
            "debug_grid.png"
        )
    )

    plt.show()

    print(f"Debug completed: {folder_name}")

# =====================================
# FOLDER MODE
# =====================================

def folder_mode(folder_name):

    input_dir = os.path.join(
        INPUT_ROOT,
        folder_name
    )

    output_dir = os.path.join(
        "data/processed/masks",
        folder_name
    )

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff"
        ))
    ])

    for file in image_files:

        image_path = os.path.join(
            input_dir,
            file
        )

        result = create_mask(image_path)

        save_path = os.path.join(
            output_dir,
            f"mask_{file}"
        )

        save_mask(
            result["mask"],
            save_path
        )

    print(f"Folder processed: {folder_name}")

# =====================================
# DATASET MODE
# =====================================

def dataset_mode():

    folders = sorted(os.listdir(INPUT_ROOT))

    for folder in folders:

        folder_path = os.path.join(
            INPUT_ROOT,
            folder
        )

        if not os.path.isdir(folder_path):
            continue

        folder_mode(folder)

    print("Dataset completed")

# =====================================
# MAIN
# =====================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    required=True,
    choices=[
        "debug",
        "folder",
        "dataset"
    ]
)

parser.add_argument(
    "--folder",
    type=str
)

args = parser.parse_args()

# =====================================
# EXECUTION
# =====================================

if args.mode == "debug":

    if args.folder is None:

        raise ValueError(
            "--folder required in debug mode"
        )

    debug_mode(args.folder)

elif args.mode == "folder":

    if args.folder is None:

        raise ValueError(
            "--folder required in folder mode"
        )

    folder_mode(args.folder)

elif args.mode == "dataset":

    dataset_mode()