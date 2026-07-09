import cv2
from ultralytics import YOLO

# ─── 1. Load the YOLO model ──────────────────────────────────────────────
# yolov8n.pt = YOLOv8 Nano, the smallest and fastest variant (~6 MB).
# It trades some accuracy for speed, which is ideal for CPU-only real-time use.
model = YOLO("yolov8n.pt")

# ─── 2. Open the webcam ──────────────────────────────────────────────────
# cv2.VideoCapture(0) opens the default webcam (index 0).
# If you have multiple cameras, try 1, 2, etc.
cap = cv2.VideoCapture(0)

cap.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, 1280) if hasattr(cv2, 'cv2') else cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, 720) if hasattr(cv2, 'cv2') else cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Safety check: make sure the camera actually opened.
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ─── 3. Main video loop ──────────────────────────────────────────────────
while True:
    # ── 3a. Read a frame from the webcam ─────────────────────────────────
    # cap.read() returns two things:
    #   ret  → True if the frame was read successfully, False otherwise.
    #   frame → The actual image as a NumPy array (H x W x 3, BGR format).
    ret, frame = cap.read()

    # If the frame failed to capture (e.g., camera disconnected), break out.
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # ── 3b. Run YOLO inference on the frame ────────────────────────────
    # model() takes the image and returns a list of Results objects.
    # We use [0] because we're processing a single image, so there's only one result.
    #   stream=True is optional but helps with memory for long videos.
    results = model(frame, classes=[0, 67], conf=0.6)[0]

    # ── 3c. Draw bounding boxes on the frame ────────────────────────────
    # results.plot() returns a new image (NumPy array) with boxes and labels drawn.
    # This is the easiest way to visualize detections without manual drawing code.
    annotated_frame = results.plot()

    # ── 3d. Display the annotated frame ────────────────────────────────
    # cv2.imshow(window_name, image) opens a window and shows the image.
    # 'YOLO Real-Time Detection' is just the title of the window.
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # ── 3e. Check for the 'q' key press ────────────────────────────────
    # cv2.waitKey(1) waits 1 millisecond for a key press and returns the ASCII code.
    # We mask with & 0xFF to ensure compatibility across systems.
    # If the pressed key is 'q' (ASCII 113), we break the loop to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── 4. Cleanup ──────────────────────────────────────────────────────────
# Release the webcam so other applications can use it.
cap.release()

# Close all OpenCV windows created by cv2.imshow().
cv2.destroyAllWindows()

print("Stream ended cleanly.")