import os
import time
import cv2
from ultralytics import YOLO
import csv 

# ─── 1. Setup Model & Camera ───────────────────────────────────────────
custom_weights_path = "runs/detect/object_tracker_runs/cpu_nano_run-2/weights/best.pt"

# Fallback to yolov8n if custom weights are not located
if os.path.exists(custom_weights_path):
    model = YOLO(custom_weights_path)
    # Target classes from your custom trained data
    TARGET_PERSON = 0
    TARGET_PHONE = 1
    print(f"Loaded custom model: {custom_weights_path}")
else:
    model = YOLO("yolov8n.pt")
    # Base COCO indices
    TARGET_PERSON = 0
    TARGET_PHONE = 67
    print("Custom weights not found; falling back to base yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set standard camera buffer and resolution to reduce CPU capture overhead
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ─── 2. CPU Performance & State Machine Configurations ─────────────────
FRAME_SKIP = 3           # Run YOLO every 3rd frame to conserve CPU cycles
INFERENCE_SIZE = 320     # 320x320 cuts operations by ~4x vs default 640x640

AWAY_THRESHOLD = 4.0     # Absence confirmed after 4 seconds with no person
PHONE_DEBOUNCE = 1.5     # Phone distraction ends only after 1.5s of absence

activity_log = []
frame_count = 0

person_last_seen = time.time()
phone_last_seen = 0.0

is_person_absent = False
phone_currently_detected = False

# Persistent detection state across skipped frames
cached_boxes = []
cached_person_in_frame = False
cached_phone_in_frame = False

# FPS tracking
prev_time = time.time()
fps = 0.0

print("🚀 CPU-Optimized Tracking started. Press 'q' to stop...")

# ─── 3. Main Loop ────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    current_time = time.time()
    frame_count += 1

    # Compute actual running display FPS
    fps = 0.9 * fps + 0.1 * (1.0 / (current_time - prev_time + 1e-6))
    prev_time = current_time

    # ── Heavy Inference Step (Skipped on intermediate frames) ──
    if frame_count % FRAME_SKIP == 0:
        cached_boxes = []
        cached_person_in_frame = False
        cached_phone_in_frame = False

        # Single forward pass at reduced resolution
        results = model(
            frame, 
            classes=[TARGET_PERSON, TARGET_PHONE], 
            conf=0.5, 
            imgsz=INFERENCE_SIZE, 
            verbose=False
        )[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            if cls_id == TARGET_PERSON:
                cached_person_in_frame = True
                cls_name = "Person"
            elif cls_id == TARGET_PHONE:
                cached_phone_in_frame = True
                cls_name = "Phone"
            else:
                cls_name = model.names.get(cls_id, f"ID {cls_id}")

            cached_boxes.append((xyxy, cls_name, conf))

    # ── 3a. Temporal Absence Logic ──
    if cached_person_in_frame:
        person_last_seen = current_time
        if is_person_absent:
            print("▶️ Person returned to desk.")
            activity_log.append({"timestamp": time.ctime(current_time), "event": "Returned to Desk"})
            is_person_absent = False
    else:
        if (current_time - person_last_seen) > AWAY_THRESHOLD and not is_person_absent:
            print("⚠️ Away! User left the desk.")
            activity_log.append({"timestamp": time.ctime(current_time), "event": "Away from Desk"})
            is_person_absent = True

    # ── 3b. Temporal Distraction Logic (Debounced) ──
    if cached_phone_in_frame:
        phone_last_seen = current_time
        if not phone_currently_detected:
            print("📱 Distraction! Phone detected.")
            activity_log.append({"timestamp": time.ctime(current_time), "event": "Phone Distraction Started"})
            phone_currently_detected = True
    else:
        if phone_currently_detected and (current_time - phone_last_seen) > PHONE_DEBOUNCE:
            print("🔒 Focused! Phone put away.")
            activity_log.append({"timestamp": time.ctime(current_time), "event": "Phone Distraction Ended"})
            phone_currently_detected = False

    # ── 3c. Lightweight Rendering ──
    for xyxy, label, conf in cached_boxes:
        box_color = (0, 0, 255) if label == "Phone" else (255, 150, 0)
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # Status Dashboard HUD
    if is_person_absent:
        status_text, status_color = "AWAY", (0, 165, 255)
    elif phone_currently_detected:
        status_text, status_color = "DISTRACTED", (0, 0, 255)
    else:
        status_text, status_color = "FOCUSED", (0, 255, 0)

    cv2.putText(frame, f"STATUS: {status_text}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Productivity Tracker (CPU)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── 4. Cleanup & Summary ────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

csv_file = "productivity_session_logs.csv"
file_exists = os.path.exists(csv_file)

if activity_log:
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "event"])
        
        # Write headers only if the file is new
        if not file_exists:
            writer.writeheader()
            
        writer.writerows(activity_log)
    
    print(f"\n📁 Activity logs successfully saved to '{csv_file}'")
else:
    print("\n⏱️ Session ended. No events recorded to log.")

print("\n=== SESSION ACTIVITY LOG ===")
for entry in activity_log:
    print(f"[{entry['timestamp']}] {entry['event']}")
print("=============================")