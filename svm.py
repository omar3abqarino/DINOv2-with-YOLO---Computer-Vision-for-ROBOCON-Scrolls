import os, pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


CROPS_DIR = "D:\\crops\\svm"                     
CLASSES = ["fake", "r1", "r2", "junk"]   
OUT_PKL = "svm_classifier.pkl"
BATCH_SIZE = 32
TEST_SIZE = 0.2                          
RANDOM_STATE = 42

device = "cuda" if torch.cuda.is_available() else "cpu"


processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)
model.eval()


def get_embeddings_batch(img_paths, batch_size=BATCH_SIZE):

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



img_paths = []
labels = []

for class_id, class_name in enumerate(CLASSES):
    class_dir = os.path.join(CROPS_DIR, class_name)

    files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"{class_name} (id={class_id}): {len(files)} crops")

    for filename in files:
        img_paths.append(os.path.join(class_dir, filename))
        labels.append(class_id)

if len(img_paths) == 0:
    raise RuntimeError("No crops found. Check CROPS_ROOT / folder names.")

labels = np.array(labels)


#L2-normalization to make sum of their squares = 1; it is easier on hardware
embeddings = get_embeddings_batch(img_paths)
embeddings = F.normalize(embeddings, p=2, dim=1).numpy()


X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels)


val_clf = SVC(kernel="linear", C=1.0, probability=True, random_state=RANDOM_STATE)
val_clf.fit(X_train, y_train)

val_preds = val_clf.predict(X_val)
print("\n--- Holdout validation report ---")
print(classification_report(y_val, val_preds, target_names=CLASSES))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_val, val_preds))
print("----------------------------------\n")



final_clf = SVC(kernel="linear", C=1.0, probability=True, random_state=RANDOM_STATE)
final_clf.fit(embeddings, labels)

with open(OUT_PKL, "wb") as f:
    pickle.dump({
        "model": final_clf,
        "classes": CLASSES,          
    }, f)

print(f"\nSaved trained SVM")
print("Done.")