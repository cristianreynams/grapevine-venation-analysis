import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# CONFIG
# =====================================

INPUT_ROOT = "data/processed/skeleton"

# Minimum connected component size
MIN_COMPONENT_SIZE = 10

# =====================================
# CORE FUNCTION
# =====================================

def prune_skeleton(image_path):

    # =====================================
    # LOAD IMAGE
    # =====================================

    skeleton = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if skeleton is None:

        raise ValueError(
            f"Could not load image: {image_path}"
        )

    # =====================================
    # CONNECTED COMPONENTS
    # =====================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton,
        connectivity=8
    )

    pruned = np.zeros_like(skeleton)

    # =====================================
    # FILTER COMPONENTS
    # =====================================

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_COMPONENT_SIZE:

            pruned[
                labels == label
            ] = 255

    return {
        "skeleton": skeleton,
        "pruned": pruned
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

    input_dir = os.path.join(
        INPUT_ROOT,
        folder_name
    )

    output_dir = os.path.join(
        "data/processed/pruned",
        "debug_img"
    )

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if (
            f.lower().endswith((
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff"
            ))
            and not f.startswith("debug")
        )
    ])[:4]

    results = []

    for file in image_files:

        image_path = os.path.join(
            input_dir,
            file
        )

        result = prune_skeleton(
            image_path
        )

        # =====================================
        # SAVE
        # =====================================

        save_path = os.path.join(
            output_dir,
            f"pruned_{file}"
        )

        save_image(
            result["pruned"],
            save_path
        )

        results.append((
            file,
            result["skeleton"],
            result["pruned"]
        ))

    # =====================================
    # VISUALIZATION
    # =====================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12,6)
    )

    axes = axes.flatten()

    for i, (name, skeleton, pruned) in enumerate(results):

        # =====================================
        # ORIGINAL SKELETON
        # =====================================

        axes[i * 2].imshow(
            skeleton,
            cmap="gray"
        )

        axes[i * 2].set_title(
            f"{name}\nSkeleton",
            fontsize=9
        )

        axes[i * 2].axis("off")

        # =====================================
        # PRUNED
        # =====================================

        axes[i * 2 + 1].imshow(
            pruned,
            cmap="gray"
        )

        axes[i * 2 + 1].set_title(
            f"{name}\nPruned",
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
        "data/processed/pruned",
        folder_name
    )

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if (
            f.lower().endswith((
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff"
            ))
            and not f.startswith("debug")
        )
    ])

    for file in image_files:

        image_path = os.path.join(
            input_dir,
            file
        )

        result = prune_skeleton(
            image_path
        )

        save_path = os.path.join(
            output_dir,
            f"pruned_{file}"
        )

        save_image(
            result["pruned"],
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