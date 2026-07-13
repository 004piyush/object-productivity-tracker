import os
from ultralytics import YOLO

def train_model():
    # 1. Use the lightweight nano weights (yolov8n.pt) 
    # This keeps training snappy and prevents your processor from bottlenecking
    model_weights = "yolov8n.pt" 
    
    print(f"🤖 Initializing architecture on CPU using: {model_weights}")
    model = YOLO(model_weights)

    # 2. Kick off the training configuration optimized for your hardware
    print("🚀 Starting training loop. Monitoring progress...")
    model.train(
        data="data.yaml",               # Points to your train/val folder mapping setup
        epochs=120,                      # 30 iterations is perfect to evaluate baseline performance
        imgsz=640,                      # Standard YOLO image resolution
        batch=4,                        # Safe batch size to prevent out-of-memory overhead on CPU
        device="cpu",                   # Directs the workload to your processor cores
        workers=0,                      # Keeps dataloading stable and sequential
        project="object_tracker_runs",  # Root directory for tracking outputs
        name="cpu_nano_run"             # Subfolder where this specific run saves metrics
    )
    
if __name__ == "__main__":
    train_model()