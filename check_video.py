
import cv2
from pathlib import Path

video_path = Path("samples") / "parking.mp4"

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video opened successfully!")
    print("Resolution:", width, "x", height)
    print("FPS:", fps)
    print("Total frames:", total_frames)
    print("Duration:", round(total_frames / fps, 2), "seconds")

    ret, frame = cap.read()
    if ret:
        cv2.imshow("Parking Video - First Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

cap.release()
