# Real-Time Object Detection & Productivity Tracker

An AI-powered application that uses computer vision to track workplace presence and manage focus analytics.

🚧 **Status:** Work in Progress — Iteratively building the core tracking pipeline.

### Current Progress:
* **Phase 1: Real-Time Inference Foundation (Done)** - Integrated OpenCV video capture with the YOLOv8n core model for real-time edge detection.
* **Phase 2: State Tracking & Logic Optimization (Done)** - Implemented custom tensor parsing to filter specific class IDs (person and cell phone) while building a 5-second temporal validation buffer and a 2-second debouncing threshold to eliminate data flickering and frame drops.
* **Phase 3: Custom Fine-Tuning Pipeline (Done)** - Overcame weak default object labeling by curating a custom dataset and executing an optimized 120-epoch training run. Successfully boosted `Cell_Phone` detection from a baseline of **34.6%** to a robust **82.8% mAP50**, and secured a highly stable **95.2%** score for `Person` detection.
