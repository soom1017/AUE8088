RUN_NAME=$1
DEVICE=$2

python utils/eval/generate_kaist_ann_json.py \
    --textListFile datasets/kaist-rgbt/custom/val.txt \
    --jsonAnnFile utils/eval/KAIST_val-B_annotation.json

python train_simple.py \
    --img 640 \
  --batch-size 16 \
  --epochs 40 \
  --data data/kaist-rgbt-custom-split.yaml \
  --cfg models/yolov11_kaist-rgbt.yaml \
  --weights yolov11s.pt \
  --workers 16 \
  --name "$RUN_NAME" \
  --device "$DEVICE" \
  --rgbt \
  --single-cls
