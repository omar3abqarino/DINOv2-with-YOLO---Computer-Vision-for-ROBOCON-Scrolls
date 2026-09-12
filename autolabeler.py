import os, json, torch, gc
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel



def get_embedding(img_path):
    #embedding of each image
    image = Image.open(img_path).convert("L").convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # equalizes the output in case one pic is larger than another
    return outputs.last_hidden_state.mean(dim=1)

def get_embeddings_batch(img_paths, batch_size=16):

    all_embeddings = []

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        #embedding of the images batch
        images = [Image.open(p).convert("L").convert("RGB") for p in batch_paths]

    
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # equalizes the output in case one pic is larger than another
        batch_embeds = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(batch_embeds.cpu())
    return torch.cat(all_embeddings, dim=0)



CROPS_DIR = "D:\\crops"
# CROPS_DIR = "crops"
METADATA_FILE = "crops_metadata.json"
# METADATA_PATH = "crops_metadata.json"
METADATA_PATH = os.path.join(CROPS_DIR, METADATA_FILE)
TEMPLATES_DIR = "new_templates"
DATASET_BASE = "Dataset"
SIMILARITY_THRESHOLD = 0.50



device = "cuda" if torch.cuda.is_available() else "cpu"


processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
model.eval()



template_embeddings = []
template_filenames = []
class_ids = []

for filename in os.listdir(TEMPLATES_DIR):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    #assign real = 0
    if "real" in filename.lower():
        class_ids.append(0)

    #assign fake = 1
    elif "fake" in filename.lower():
        class_ids.append(1)

    else:
        print("Neither real nor fake. :)")
        print(f"Skipping {filename}")
        continue


    filepath = os.path.join(TEMPLATES_DIR, filename)
    template_embeddings.append(get_embedding(filepath))
    template_filenames.append(filename)



template_tensor = torch.cat(template_embeddings, dim = 0).to(device)
template_tensor = F.normalize(template_tensor, p=2, dim=1)
class_tensor = torch.tensor(class_ids).to(device)


with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)


for img_name, data in metadata.items():
    og_width = data["img_w"]
    og_height = data["img_h"]
    split = data.get("split", "train")
    yolo_lines = []

    # keep only boxes whose crop actually exists on disk, in lockstep with crop_paths
    valid_boxes = []
    crop_paths = []
    for box in data["boxes"]:
        CROP_PATH = os.path.join(CROPS_DIR, box["crop_filename"])
        if not os.path.exists(CROP_PATH):
            print(f"  [MISSING] {box['crop_filename']} not found, skipping")
            continue
        valid_boxes.append(box)
        crop_paths.append(CROP_PATH)

    if not crop_paths:
        print("Done labeling for this image :)")
        continue


    #L2-normalization to make sum of their squares = 1; it is easier on hardware
    crop_embeddings = get_embeddings_batch(crop_paths)
    crop_embeddings = F.normalize(crop_embeddings.to(device), p=2, dim=1)

    #continuing the cosine similarity by just doing the dot product
    similarities = crop_embeddings @ template_tensor.T

    best_scores, best_match_idxs = torch.max(similarities, dim=1)


    for box, best_score, best_match_idx in zip(valid_boxes, best_scores, best_match_idxs):
        best_score = best_score.item()
        best_idx = best_match_idx.item()

        if best_score < SIMILARITY_THRESHOLD:
            print(f"  [REJECTED] {box['crop_filename']} Not a valid scroll (Score: {best_score:.3f})")
            continue  # skips because it is neither real nor fake

        best_class_id = class_tensor[best_idx].item()
        matched_template = template_filenames[best_idx]
        print(f"  [ACCEPTED] {box['crop_filename']} -> {matched_template} (Score: {best_score:.3f})")

        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        cx = ((x1 + x2) / 2) / og_width
        cy = ((y1 + y2) / 2) / og_height
        w = (x2 - x1) / og_width
        h = (y2 - y1) / og_height



        yolo_lines.append(f"{best_class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")


    if yolo_lines:
        labels_dir = os.path.join(DATASET_BASE, split, "labels")
        os.makedirs(labels_dir, exist_ok=True)

        arr = img_name.split(".", -1)[:-1]
        new_name = ".".join(arr) + ".txt"

        with open(os.path.join(labels_dir, new_name), "w") as f:
            f.write("\n".join(yolo_lines))

    print("Done labeling for this image :)")

    #free memory
    gc.collect()
    torch.cuda.empty_cache()


print("Donnnnneeeee")