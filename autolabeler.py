import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel



def get_embedding(img_path):
    #embedding of each image
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # equalizes the output in case one pic is larger than another
    return outputs.last_hidden_state.mean(dim=1)



# CROPS_DIR = "D:\\crops"
CROPS_DIR = "crops"
METADATA_FILE = "crops_metadata.json"
METADATA_PATH = "crops_metadata.json"
# METADATA_PATH = os.path.join(CROPS_DIR, METADATA_FILE)
TEMPLATES_DIR = "templates"
LABELS_DIR = "labels"
SIMILARITY_THRESHOLD = 0.65

os.makedirs(LABELS_DIR, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"


processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
model.eval()



template_embeddings = []
class_ids = []

for i in range(1, 16):
    #load real = 0
    template_embeddings.append(get_embedding(f"{TEMPLATES_DIR}\\real_{i}.jpg"))
    class_ids.append(0)

    #load fake = 1
    template_embeddings.append(get_embedding(f"{TEMPLATES_DIR}\\fake_{i}.jpg"))
    class_ids.append(1)


template_tensor = F.normalize(torch.cat(template_embeddings, dim = 0).to(device), p=2, dim=1)
class_tensor = torch.tensor(class_ids).to(device)


with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)


for img_name, data in metadata.items():
    og_width = data["img_w"]
    og_height = data["img_h"]
    yolo_lines = []


    for box in data["boxes"]:
        CROP_PATH = os.path.join(CROPS_DIR, box["crop_filename"])
        if not os.path.exists(CROP_PATH):
            continue

        #L2-normalization to make sum of their squares = 1; it is easier on hardware
        crop_embedding = F.normalize(get_embedding(CROP_PATH), p=2, dim=1)

        #continuing the cosine similarity by just doing the dot product
        similarities = crop_embedding @ template_tensor.T


        best_score, best_match_idx = torch.max(similarities, dim=1)
        best_score = best_score.item()

        if best_score < SIMILARITY_THRESHOLD:
            print(f"  [REJECTED] Not a valid scroll (Score: {best_score:.3f})")
            continue  # skips because it is neither real nor fake



        best_class_id = class_tensor[best_match_idx.item()].item()

        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        cx = ((x1 + x2) / 2) / og_width
        cy = ((y1 + y2) / 2) / og_height
        w = (x2 - x1) / og_width
        h = (y2 - y1) / og_height



        yolo_lines.append(f"{best_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")


    if yolo_lines:
        arr = img_name.split(".", -1)[:-1]
        new_name = ".".join(arr) + ".txt"

        with open(os.path.join(LABELS_DIR, new_name), "w") as f:
            f.write("\n".join(yolo_lines))

    print("Done labeling for this image :)")


print("Donnnnneeeee")