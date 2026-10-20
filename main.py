
import cv2
import json
import pickle
from pathlib import Path
from skimage.transform import resize

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "samples" / "parking.mp4"
SLOTS_FILE = ROOT / "parking_spaces.json"
MODEL_FILE = ROOT / "model.p"

WIDTH, HEIGHT = 1280, 720

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(SLOTS_FILE, "r") as f:
    slots = json.load(f)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Could not open samples/parking.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    empty_count = 0
    occupied_count = 0

    for i, (x1, y1, x2, y2) in enumerate(slots, start=1):
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = resize(crop, (15, 15, 3))
        features = crop.flatten().reshape(1, -1)
        prediction = int(model.predict(features)[0])

        if prediction == 0:
            empty_count += 1
            color = (0, 255, 0)
            label = "EMPTY"
        else:
            occupied_count += 1
            color = (0, 0, 255)
            label = "OCCUPIED"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"{i}: {label}", (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    total_spaces = len(slots)
    occupancy = (
        (occupied_count / total_spaces) * 100
        if total_spaces > 0 else 0
    )

    cv2.rectangle(frame, (10, 10), (390, 155), (30, 30, 30), -1)

    cv2.putText(frame, f"Total Spaces: {total_spaces}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Available: {empty_count}", (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Occupied: {occupied_count}", (20, 106),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Occupancy: {occupancy:.1f}%", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Parking Space Counter", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
