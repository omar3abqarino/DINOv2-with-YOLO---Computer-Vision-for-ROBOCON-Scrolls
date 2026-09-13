from ultralytics import YOLO
import csv, os, time

DATASET_YAML_PATH = "C:\\Users\\a\\Desktop\\Coding\\M.I.A\\DINOv2-with-YOLO---Computer-Vision-for-ROBOCON-Scrolls\\Dataset\\dataset.yaml"
# SCALES = ["n", "s", "m", "l"]
SCALES = ["n"]
EPOCHS = 100
IMGSZ = 640
BATCH = -1
PATIENCE = 10          # early stopping if no improvement for N epochs
RESULTS_CSV = "yolo26_scale_comparison.csv"
DEGREES = 20.0           # random rotation +/- degrees per training image
PERSPECTIVE = 0.0008     # random perspective warp
MULTI_SCALE = 0.5 

def extract_metrics(val_metrics):
    box = val_metrics.box

    overall = {
        "precision": round(float(box.mp), 4),        # mean precision across classes
        "recall": round(float(box.mr), 4),            # mean recall across classes
        "mAP50": round(float(box.map50), 4),
        "mAP75": round(float(box.map75), 4),
        "mAP50-95": round(float(box.map), 4),
        "fitness": round(float(val_metrics.fitness), 4),
    }

    # Ultralytics' own internal timing from the val run (separate signal
    # from our own end-to-end latency benchmark further down)
    speed = {f"val_{k}_ms": round(v, 3) for k, v in val_metrics.speed.items()}

    return overall, speed


def main():
    results_summary = []

    for scale in SCALES:
        model_name = f"yolo26{scale}.pt"
        run_name = f"yolo26{scale}_run"
        print(f"\n{'=' * 50}\nTraining {model_name}\n{'=' * 50}")

        model = YOLO(model_name)

        train_start = time.time()
        model.train(
            data=DATASET_YAML_PATH,
            epochs=EPOCHS,
            device=0,
            workers=6,     # 8GB Ram LIMIT
            imgsz=IMGSZ,
            batch=BATCH,   # Your 10GB VRAM can comfortably handle standard batch sizes
            patience=PATIENCE,
            name=run_name,
            cache=False,    # CRITICAL: Prevents YOLO from loading images into RAM
            degrees=DEGREES,
            perspective=PERSPECTIVE,
            multi_scale=MULTI_SCALE,
        )
        train_time_min = (time.time() - train_start) / 60

        # Reload the best checkpoint explicitly rather than trusting
        # in-memory state, so metrics reflect the best epoch, not the last.
        best_weights = os.path.join("runs", "detect", run_name, "weights", "best.pt")
        best_model = YOLO(best_weights)

        val_metrics = best_model.val(data=DATASET_YAML_PATH)
        overall, val_speed = extract_metrics(val_metrics)



        row = {
            "scale": f"yolo26{scale}",
            **overall,
            **val_speed,
            "train_time_min": round(train_time_min, 2),
            "weights_path": best_weights,
        }

        results_summary.append(row)

        print(f"\n--- {model_name} summary ---")
        for k, v in row.items():
            print(f"  {k}: {v}")

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_summary[0].keys())
        writer.writeheader()
        writer.writerows(results_summary)

    print(f"\n{'=' * 50}\nAll scales trained.\n{'=' * 50}")
    for row in results_summary:
        print(row)


if __name__ == "__main__":
    main()