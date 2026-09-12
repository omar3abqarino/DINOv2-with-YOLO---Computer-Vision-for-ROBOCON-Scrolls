import os, json, time, gc, torch
import cv2 as cv
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology


TRAIN_DIR = "Dataset\\train\\images"
VAL_DIR = "Dataset\\val\\images"
TEST_DIR = "Dataset\\test\\images"
INPUT_DIRS = [TRAIN_DIR, VAL_DIR, TEST_DIR]

CROPS_DIR = "D:\\crops"
METADATA_FILE = "crops_metadata.json"
METADATA_PATH = os.path.join(CROPS_DIR, METADATA_FILE)
PADDING = 7

os.makedirs(CROPS_DIR, exist_ok=True)


box_detector = GroundingDINO(ontology=CaptionOntology({"cardboard box": "box"}))



print("Scanning images for boxes.")

start = time.time()
count = 0
metadata = {}


for INPUT_DIR in INPUT_DIRS:
    if not os.path.exists(INPUT_DIR):
        print("Skipping non-existent directory.")
        continue

    split = os.path.basename(os.path.dirname(INPUT_DIR))
    target_crop_dir = os.path.join(CROPS_DIR, split)
    os.makedirs(target_crop_dir, exist_ok=True)

    for img_name in os.listdir(INPUT_DIR):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')): 
            continue
            
        img_path = os.path.join(INPUT_DIR, img_name)
        try:
            image_cv = cv.imread(img_path)
        except:
            print(f"Pic failed {img_name}")
        if image_cv is None:
            continue
            
        img_h, img_w, _ = image_cv.shape
        
        try:
            detections = box_detector.predict(img_path)
        except:
            print(f"Pic failed {img_name}")
        
        metadata[img_name] = {
            "img_w": img_w,
            "img_h": img_h,
            "split": split,
            "boxes": []
        }
        
        
        for idx, xyxy in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, xyxy)
            
            
            x1, y1 = max(0, x1 - PADDING), max(0, y1 - PADDING)
            x2, y2 = min(img_w, x2 + PADDING), min(img_h, y2 + PADDING)
            
            crop_cv = image_cv[y1:y2, x1:x2]
            
            
            if crop_cv.size == 0: 
                continue

            arr = img_name.split(".", -1)[:-1]

            new_name = ".".join(arr)
            
            crop_filename = f"{new_name}_crop_{idx}.jpg"
            crop_path = os.path.join(target_crop_dir, crop_filename)
            
            
            cv.imwrite(crop_path, crop_cv)
            
            rel_crop_path = os.path.join(split, crop_filename)
            metadata[img_name]["boxes"].append({"crop_filename": rel_crop_path, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            
        count += 1


        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
    print(f"{INPUT_DIR} is Done")

    
end = time.time()


print("Done")
if count > 0:
    print(f"avg per img: {(end-start)/count} seconds")