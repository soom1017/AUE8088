RUN_NAME=$1
DEVICE=$2

python val_simple.py \
    --weights "runs/train/$RUN_NAME/weights/best.pt" \
    --data data/kaist-rgbt.yaml \
    --device "$DEVICE" \
    --name "$RUN_NAME" \
    --task test \
    --verbose \
    --single-cls \
    --rgbt