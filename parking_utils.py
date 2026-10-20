
import pickle
from pathlib import Path

import cv2
import numpy as np
from skimage.transform import resize


MODEL_PATH = Path("model.p")


def load_model():
    """Load the previously trained parking-space classifier."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.p not found. Download the model from the "
            "original project's linked Google Drive folder."
        )

    # Only load pickle files from a trusted source.
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def extract_parking_spaces(mask):
    """Extract one bounding box per connected mask component."""
    if mask is None:
        raise ValueError("Could not read the parking-space mask.")

    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    spaces = []

    for label in range(1, count):
        x, y, width, height, area = stats[label]

        if width > 0 and height > 0 and area > 0:
            spaces.append((int(x), int(y), int(width), int(height)))

    if not spaces:
        raise ValueError(
            "No parking spaces found. Check the mask image."
        )

    return spaces


def classify_space(model, crop):
    """Predict whether a parking-space crop is empty."""
    if crop.size == 0:
        return False

    resized = resize(
        crop, (15, 15, 3), anti_aliasing=True
    )
    features = resized.flatten().reshape(1, -1)

    prediction = int(np.asarray(model.predict(features)).ravel()[0])

    # This mapping follows the original project's convention:
    # class 0 = empty, other class = occupied.
    return prediction == 0
