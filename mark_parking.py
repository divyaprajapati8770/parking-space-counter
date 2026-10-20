
import cv2
import json

VIDEO = "samples/parking.mp4"
OUTPUT = "parking_spaces.json"
WIDTH, HEIGHT = 1280, 720

cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read the video.")

frame = cv2.resize(frame, (WIDTH, HEIGHT))
display = frame.copy()
start_point = None
rectangles = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    global start_point, drawing, display

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        display = frame.copy()
        for x1, y1, x2, y2 in rectangles:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(display, start_point, (x, y), (0, 255, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        x1, y1 = start_point
        x2, y2 = x, y
        rectangles.append([
            min(x1, x2), min(y1, y2),
            max(x1, x2), max(y1, y2)
        ])
        drawing = False
        display = frame.copy()
        for i, (a, b, c, d) in enumerate(rectangles, 1):
            cv2.rectangle(display, (a, b), (c, d), (0, 255, 0), 2)
            cv2.putText(display, str(i), (a, b - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.namedWindow("Select Parking Slots")
cv2.setMouseCallback("Select Parking Slots", mouse_callback)

print("Drag the mouse to draw a rectangle around each parking slot.")
print("Press S to save, U to undo the last rectangle, Q to quit.")

while True:
    cv2.imshow("Select Parking Slots", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        with open(OUTPUT, "w") as f:
            json.dump(rectangles, f)
        print(f"Saved {len(rectangles)} parking slots to {OUTPUT}")
        break

    elif key == ord("u") and rectangles:
        rectangles.pop()
        display = frame.copy()
        for i, (a, b, c, d) in enumerate(rectangles, 1):
            cv2.rectangle(display, (a, b), (c, d), (0, 255, 0), 2)
            cv2.putText(display, str(i), (a, b - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    elif key == ord("q"):
        break

cv2.destroyAllWindows()

