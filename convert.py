import json
import os

# 1. The exact order of your classes for your YOLO model mapping
classes = [
    "Person", "Cell_Phone", "Calculator", "SwitchBoard", "Cable", "Ring",
    "Stylus", "PenTab", "Cooling Pad", "Dumbell", "Bag", "Earbuds",
    "WhiteBox", "Bottle", "Teddy", "Remote", "Charger", "Cable(White)"
]
class_to_id = {cls: idx for idx, cls in enumerate(classes)}

# Load your Label Studio export data
with open("result.json", "r") as f:
    data = json.load(f)

# Create output folder for the YOLO labels
os.makedirs("labels", exist_ok=True)

for task in data:
    if not task.get("annotations"):
        continue
        
    image_name = os.path.basename(task["file_upload"])
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    annotations = task["annotations"][0]["result"]
    
    # Group bounding box shapes with their corresponding label text by ID
    regions = {}
    for res in annotations:
        reg_id = res["id"]
        if reg_id not in regions:
            regions[reg_id] = {"box": None, "class": None}
            
        if res["type"] == "rectangle":
            regions[reg_id]["box"] = res["value"]
        elif res["type"] == "taxonomy":
            # Extract out the string text from the nested taxonomy format
            if "taxonomy" in res["value"] and res["value"]["taxonomy"]:
                regions[reg_id]["class"] = res["value"]["taxonomy"][0][0]

    # Convert values to standard normalized YOLO formatting (0.0 to 1.0 range)
    with open(os.path.join("labels", txt_name), "w") as out_f:
        for reg_id, content in regions.items():
            box = content["box"]
            cls_name = content["class"]
            
            if box and cls_name in class_to_id:
                cls_id = class_to_id[cls_name]
                
                # Convert center coordinates from 0-100 percentage scale to 0-1
                x_center = (box["x"] + box["width"] / 2) / 100.0
                y_center = (box["y"] + box["height"] / 2) / 100.0
                w = box["width"] / 100.0
                h = box["height"] / 100.0
                
                out_f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("Finished successfully! Check your new '/labels' folder for the YOLO data.")