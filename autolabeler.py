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



METADATA_FILE = "crops_metadata.json"
CROPS_DIR = "D:\\crops"
TEMPLATES_DIR = "templates"
LABELS_DIR = "labels"

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


template_tensor = torch.cat(template_embeddings, dim = 0).to(device)
class_tensor = torch.tensor(class_ids).to(device)


