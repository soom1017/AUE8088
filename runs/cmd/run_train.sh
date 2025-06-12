RUN_NAME=$1
DEVICE=$2

python utils/eval/generate_kaist_ann_json.py \
    --textListFile datasets/kaist-rgbt/train-all-04.txt

echo "====== json file generated for training ======"

python train_simple.py \
    --img 640 \
  --batch-size 16 \
  --epochs 20 \
  --data data/kaist-rgbt.yaml \
  --cfg models/yolov5n_kaist-rgbt.yaml \
  --weights yolov5n.pt \
  --workers 16 \
  --name "$RUN_NAME" \
  --device "$DEVICE" \
  --rgbt