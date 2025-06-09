import random
import os
from pathlib import Path

random.seed(42)

INPUT_PATH = "datasets/kaist-rgbt/train-all-04.txt"
OUTPUT_DIR = Path("datasets/kaist-rgbt/folds")
num_folds = 5

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

random.shuffle(lines)

fold_size = len(lines) // num_folds
for i in range(num_folds):
    start = i * fold_size
    end = (i + 1) * fold_size if i < num_folds - 1 else len(lines)
    val_lines = lines[start:end]
    train_lines = lines[:start] + lines[end:]

    fold_path = OUTPUT_DIR / f"fold{i}"
    os.makedirs(fold_path, exist_ok=True)

    with open(fold_path / "val.txt", "w", encoding="utf-8") as f_val:
        f_val.writelines(val_lines)
    with open(fold_path / "train.txt", "w", encoding="utf-8") as f_train:
        f_train.writelines(train_lines)

    print(f"Fold {i+1}: {len(train_lines)} train, {len(val_lines)} val lines saved to {fold_path}")