import json
import os
import re

# 1. Exact order of classes for YOLO model mapping
classes = [
    "Person", "Cell_Phone", "Calculator", "SwitchBoard", "Cable", "Ring",
    "Stylus", "PenTab", "Cooling Pad", "Dumbell", "Bag", "Earbuds",
    "WhiteBox", "Bottle", "Teddy", "Remote", "Charger", "Cable(White)"
]
class_to_id = {cls: idx for idx, cls in enumerate(classes)}

# 2. Load Label Studio export
json_path = "result.json"
if not os.path.exists(json_path):
    print(f"Error: '{json_path}' not found.")
    exit()

with open(json_path, "r") as f:
    data = json.load(f)

output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)
converted_count = 0

# 3. Parse tasks and convert coordinates
for task in data:
    if not task.get("annotations"):
        continue

    # Strip Label Studio random hash prefix (e.g. 'a1b2c3d4-img.jpg' -> 'img.jpg')
    raw_filename = os.path.basename(task.get("file_upload", ""))
    clean_filename = re.sub(r'^[a-f0-9]+-', '', raw_filename)
    txt_name = os.path.splitext(clean_filename)[0] + ".txt"

    # Group box coordinates with taxonomy labels by ID
    regions = {}
    for annotation in task["annotations"]:
        for res in annotation.get("result", []):
            reg_id = res.get("id")
            if not reg_id:
                continue
            if reg_id not in regions:
                regions[reg_id] = {"box": None, "class": None}

            if res.get("type") == "rectangle":
                regions[reg_id]["box"] = res.get("value")
            elif res.get("type") == "taxonomy":
                tax_val = res.get("value", {}).get("taxonomy", [])
                if tax_val and len(tax_val[0]) > 0:
                    regions[reg_id]["class"] = tax_val[0][0]

    # Write normalized YOLO txt file
    out_path = os.path.join(output_dir, txt_name)
    with open(out_path, "w") as out_f:
        for reg_id, content in regions.items():
            box = content["box"]
            cls_name = content["class"]

            if box and cls_name in class_to_id:
                cls_id = class_to_id[cls_name]

                # Convert center coordinates from 0-100 scale to normalized 0-1 scale
                x_center = (box["x"] + box["width"] / 2.0) / 100.0
                y_center = (box["y"] + box["height"] / 2.0) / 100.0
                w = box["width"] / 100.0
                h = box["height"] / 100.0

                out_f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    converted_count += 1

print(f"Finished successfully! Converted {converted_count} files into '/{output_dir}'.")