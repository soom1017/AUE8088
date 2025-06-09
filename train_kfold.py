from pathlib import Path
import subprocess

FOLD_YAML_DIR = Path("data/folds")
FOLD_YAML_DIR.mkdir(exist_ok=True)

FOLD_VAL_TXT_DIR = Path("datasets/kaist-rgbt/folds")

for i in range(5):
    # Generate KAIST_annotation.json for each fold's validation set
    cmd = [
        "python", "utils/eval/generate_kaist_ann_json.py",
        "--textListFile", str(FOLD_VAL_TXT_DIR / f"fold{i}/val.txt"),
    ]
    print(f"=== Generating KAIST_annotation.json for Fold {i} ===")
    subprocess.run(cmd)

    # Train YOLOv5 on each fold
    cmd = [
        "python", "train_simple.py",
        "--device", "1",
        "--img", "640",
        "--batch-size", "16",
        "--epochs", "20",
        "--data", str(FOLD_YAML_DIR / f"fold{i}.yaml"),
        "--cfg", "models/yolov5n_kaist-rgbt.yaml",
        "--weights", "yolov5n.pt",
        "--workers", "16",
        "--name", f"yolov5n-rgbt-fold{i}",
        "--rgbt",
        "--single-cls",
    ]
    print(f"=== Fold {i} Training ===")
    subprocess.run(cmd)