sh distributed_train.sh 8 /data/ImageNet \
  --model volo_h12_l18 --img-size 224 \
  -b 128 --lr 1.6e-3 --drop-path 0.1 --apex-amp \
  --token-label --token-label-size 14 --token-label-data /path/to/token_label_data \
  --model-ema --model-ema-decay 0.998 0.9986 0.999 0.9996 --epochs 100
