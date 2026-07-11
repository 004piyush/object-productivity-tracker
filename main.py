import cv2
import time
from ultralytics import YOLO

# ─── 1. Setup Model & Camera ─────────────────────────────────────────────
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ─── 2. Phase 2 Tracking Variables ───────────────────────────────────────
# This list will store all our logged events as dictionaries
activity_log = []

# Time tracking states
person_last_seen = time.time()
is_person_absent = False
phone_currently_detected = False

print("Tracking started. Press 'q' to stop and view logs...")

# ─── 3. Main Loop ────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    current_time = time.time()
    
    # Run inference filtered ONLY for Person (0) and Cell Phone (67)
    results = model(frame, classes=[0, 67], conf=0.6)[0]
    annotated_frame = results.plot()

    # Track what is visible in the current frame
    person_in_frame = False
    phone_in_frame = False

    # Inspect the bounding boxes found by YOLO
    for box in results.boxes:
        cls_id = int(box.cls[0]) # Get the class ID (0 or 67)
        
        if cls_id == 0:
            person_in_frame = True
        elif cls_id == 67:
            phone_in_frame = True

    # ─── 3a. ABSENCE LOGIC (5-Second Buffer) ─────────────────────────────
    if person_in_frame:
        person_last_seen = current_time
        if is_person_absent:
            # You just came back! Log the return
            print("▶️ Welcome back! Person detected again.")
            activity_log.append({"timestamp": time.ctime(current_time), "event": "Returned to Desk"})
            is_person_absent = False
    else:
        # Person is missing. Check if they've been gone for > 5 seconds
        time_gone = current_time - person_last_seen
        if time_gone > 5 and not is_person_absent:
            print("⚠️ Away! No person detected for more than 5 seconds.")
            activity_log.append({"timestamp": time.ctime(current_time), "event": "Away from Desk"})
            is_person_absent = True

    # ─── 3b. DISTRACTION LOGIC ───────────────────────────────────────────
    if phone_in_frame and not phone_currently_detected:
        print("📱 Distraction! Cell phone usage detected.")
        activity_log.append({"timestamp": time.ctime(current_time), "event": "Phone Distraction Started"})
        phone_currently_detected = True
        
    elif not phone_in_frame and phone_currently_detected:
        print("🔒 Focused! Phone put away.")
        activity_log.append({"timestamp": time.ctime(current_time), "event": "Phone Distraction Ended"})
        phone_currently_detected = False

    # Display live video
    cv2.imshow("Productivity Tracker - Phase 2", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ─── 4. Cleanup & Print Log ──────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

print("\n=== SESSION ACTIVITY LOG ===")
for entry in activity_log:
    print(f"[{entry['timestamp']}] {entry['event']}")
print("=============================")