import cv2
import os
import argparse
import numpy as np
import pandas as pd

from skimage.measure import label, regionprops

# =====================================
# CONFIG
# =====================================

INPUT_ROOT = "data/processed/skeleton"

OUTPUT_CSV = "outputs/features/features.csv"

# =====================================
# FEATURE FUNCTIONS
# =====================================

def count_endpoints(skeleton):

    endpoints = 0

    padded = np.pad(
        skeleton,
        ((1,1),(1,1)),
        mode="constant"
    )

    rows, cols = skeleton.shape

    for y in range(1, rows + 1):

        for x in range(1, cols + 1):

            if padded[y, x] == 0:
                continue

            neighborhood = padded[
                y-1:y+2,
                x-1:x+2
            ]

            neighbors = np.sum(
                neighborhood
            ) - 1

            if neighbors == 1:
                endpoints += 1

    return endpoints

# =====================================

def count_junctions(skeleton):

    junctions = 0

    padded = np.pad(
        skeleton,
        ((1,1),(1,1)),
        mode="constant"
    )

    rows, cols = skeleton.shape

    for y in range(1, rows + 1):

        for x in range(1, cols + 1):

            if padded[y, x] == 0:
                continue

            neighborhood = padded[
                y-1:y+2,
                x-1:x+2
            ]

            neighbors = np.sum(
                neighborhood
            ) - 1

            if neighbors >= 3:
                junctions += 1

    return junctions

# =====================================

def extract_features(image_path):

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
    # BOOLEAN
    # =====================================

    binary = skeleton > 0

    # =====================================
    # BASIC FEATURES
    # =====================================

    total_pixels = np.sum(binary)

    height, width = binary.shape

    image_area = height * width

    density = total_pixels / image_area

    # =====================================
    # CONNECTED COMPONENTS
    # =====================================

    labeled = label(binary)

    regions = regionprops(labeled)

    component_count = len(regions)

    # =====================================
    # ENDPOINTS / JUNCTIONS
    # =====================================

    endpoints = count_endpoints(
        binary.astype(np.uint8)
    )

    junctions = count_junctions(
        binary.astype(np.uint8)
    )

    # =====================================
    # LONGEST COMPONENT
    # =====================================

    max_component_area = 0

    for region in regions:

        if region.area > max_component_area:

            max_component_area = region.area

    # =====================================
    # FEATURES DICTIONARY
    # =====================================

    features = {

        "filename":
            os.path.basename(image_path),

        "skeleton_pixels":
            int(total_pixels),

        "density":
            float(density),

        "components":
            int(component_count),

        "endpoints":
            int(endpoints),

        "junctions":
            int(junctions),

        "largest_component":
            int(max_component_area),

        "height":
            int(height),

        "width":
            int(width)
    }

    return features

# =====================================
# SAVE CSV
# =====================================

def save_csv(features_list):

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    df = pd.DataFrame(
        features_list
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

# =====================================
# PROCESS FOLDER
# =====================================

def process_folder(folder_name):

    input_dir = os.path.join(
        INPUT_ROOT,
        folder_name
    )

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

    features_list = []

    for file in image_files:

        image_path = os.path.join(
            input_dir,
            file
        )

        features = extract_features(
            image_path
        )

        features["folder"] = folder_name

        features_list.append(
            features
        )

    return features_list

# =====================================
# DATASET MODE
# =====================================

def dataset_mode():

    all_features = []

    folders = sorted(os.listdir(INPUT_ROOT))

    for folder in folders:

        folder_path = os.path.join(
            INPUT_ROOT,
            folder
        )

        if not os.path.isdir(folder_path):
            continue

        folder_features = process_folder(
            folder
        )

        all_features.extend(
            folder_features
        )

        print(f"Processed: {folder}")

    save_csv(all_features)

    print(
        f"Features saved to: {OUTPUT_CSV}"
    )

# =====================================
# FOLDER MODE
# =====================================

def folder_mode(folder_name):

    features = process_folder(
        folder_name
    )

    save_csv(features)

    print(
        f"Features saved to: {OUTPUT_CSV}"
    )

# =====================================
# MAIN
# =====================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    required=True,
    choices=[
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

if args.mode == "folder":

    if args.folder is None:

        raise ValueError(
            "--folder required in folder mode"
        )

    folder_mode(args.folder)

elif args.mode == "dataset":

    dataset_mode()