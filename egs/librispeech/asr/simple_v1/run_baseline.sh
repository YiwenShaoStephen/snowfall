#!/usr/bin/env bash

set -eou pipefail

stage=1
eps=0.001
exp_dir=exp/baseline-full
mkdir -p $exp_dir  
if [ $stage -le 1 ]; then
  CUDA_VISIBLE_DEVICES=$(free-gpu) python3 ./mmi_att_transformer_train_adv.py \
  		      --world-size 1 \
  		      --raw true \
		      --full-libri true \
  		      --num-epochs 20 \
                      --max-duration 120 \
  		      --exp $exp_dir | tee -a $exp_dir/log.txt
fi
exit 0
if [ $stage -le 2 ]; then
  CUDA_VISIBLE_DEVICES=$(free-gpu) python3 ./mmi_att_transformer_decode_adv.py \
		      --epoch 3 \
		      --avg 1 \
		      --max-duration 100 \
		      --exp $exp_dir \
		      --dataset train
fi
