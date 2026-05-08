import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# CONFIG
# =====================================

IMAGE_ROOT = "data/processed/cropped"
MASK_ROOT = "data/processed/masks"

# =====================================
# CORE FUNCTION
# =====================================

def extract_leaf(image_path, mask_path):

    # =====================================
    # LOAD IMAGE
    # =====================================

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    # =====================================
    # LOAD MASK
    # =====================================

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise ValueError(
            f"Could not load mask: {mask_path}"
        )

    # =====================================
    # APPLY MASK
    # =====================================

    kernel_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3,3)
    )

    mask = cv2.erode(
        mask,
        kernel_erode,
        iterations=1
    )

    isolated = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return {
        "original": image,
        "mask": mask,
        "isolated": isolated
    }

# =====================================
# SAVE FUNCTION
# =====================================

def save_image(image, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    cv2.imwrite(path, image)

# =====================================
# DEBUG MODE
# =====================================

def debug_mode(folder_name):

    image_dir = os.path.join(
        IMAGE_ROOT,
        folder_name
    )

    mask_dir = os.path.join(
        MASK_ROOT,
        folder_name
    )

    output_dir = os.path.join(
        "data/processed/isolated",
        "debug_img"
    )

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(image_dir)
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
            image_dir,
            file
        )

        mask_path = os.path.join(
            mask_dir,
            f"mask_{file}"
        )

        result = extract_leaf(
            image_path,
            mask_path
        )

        # =====================================
        # SAVE
        # =====================================

        save_path = os.path.join(
            output_dir,
            f"isolated_{file}"
        )

        save_image(
            result["isolated"],
            save_path
        )

        isolated_rgb = cv2.cvtColor(
            result["isolated"],
            cv2.COLOR_BGR2RGB
        )

        results.append((
            file,
            isolated_rgb
        ))

    # =====================================
    # VISUALIZATION
    # =====================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8,8)
    )

    axes = axes.flatten()

    for ax, (name, image) in zip(axes, results):

        ax.imshow(image)

        ax.set_title(
            name,
            fontsize=9
        )

        ax.axis("off")

    plt.tight_layout()

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

    image_dir = os.path.join(
        IMAGE_ROOT,
        folder_name
    )

    mask_dir = os.path.join(
        MASK_ROOT,
        folder_name
    )

    output_dir = os.path.join(
        "data/processed/isolated",
        folder_name
    )

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(image_dir)
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
            image_dir,
            file
        )

        mask_path = os.path.join(
            mask_dir,
            f"mask_{file}"
        )

        result = extract_leaf(
            image_path,
            mask_path
        )

        save_path = os.path.join(
            output_dir,
            f"isolated_{file}"
        )

        save_image(
            result["isolated"],
            save_path
        )

    print(f"Folder processed: {folder_name}")

# =====================================
# DATASET MODE
# =====================================

def dataset_mode():

    folders = sorted(os.listdir(IMAGE_ROOT))

    for folder in folders:

        folder_path = os.path.join(
            IMAGE_ROOT,
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