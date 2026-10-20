
import pickle
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.transform import resize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "clf-data" / "clf-data"

with open(ROOT / "model.p", "rb") as file:
    model = pickle.load(file)

empty_images = list((DATA / "empty").glob("*"))
non_empty_images = list((DATA / "not_empty").glob("*"))

empty_images = [
    p for p in empty_images
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]
non_empty_images = [
    p for p in non_empty_images
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

print("Empty images:", len(empty_images))
print("Non-empty images:", len(non_empty_images))


def predict_image(path):
    image = imread(str(path))

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[:, :, :3]

    image = resize(image, (15, 15, 3), anti_aliasing=True)
    features = image.flatten().reshape(1, -1)

    prediction = model.predict(features)
    print(f"{path.name} -> Predicted class: {prediction[0]}")


if empty_images:
    predict_image(empty_images[0])

if non_empty_images:
    predict_image(non_empty_images[0])

    print("Model classes:", model.classes_)
print("Model type:", type(model).__name__)
