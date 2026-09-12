import cv2 as cv
import os
import glob

INPUT_DIR = "templates"
OUTPUT_DIR = "new_templates"

os.makedirs(OUTPUT_DIR, exist_ok=True)
#to search for the images
files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))


for filepath in files:
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    
    img = cv.imread(filepath)
    if img is None:
        continue

    
    variants = {
        f"{name}_norm": img,
        f"{name}_flip": cv.flip(img, 1)
    }

    # total possibilities = 30 * 4 * 2 = 240 template
    for variant_name, mat in variants.items():
        cv.imwrite(os.path.join(OUTPUT_DIR, f"{variant_name}_0{ext}"), mat)
        cv.imwrite(os.path.join(OUTPUT_DIR, f"{variant_name}_90{ext}"), cv.rotate(mat, cv.ROTATE_90_CLOCKWISE))
        cv.imwrite(os.path.join(OUTPUT_DIR, f"{variant_name}_180{ext}"), cv.rotate(mat, cv.ROTATE_180))
        cv.imwrite(os.path.join(OUTPUT_DIR, f"{variant_name}_270{ext}"), cv.rotate(mat, cv.ROTATE_90_COUNTERCLOCKWISE))

print(f"Done Template Variations!")