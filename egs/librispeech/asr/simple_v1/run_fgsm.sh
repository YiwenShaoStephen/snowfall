#!/usr/bin/env bash

# on clsp grid
. ./utils/cmd.sh
. path.sh

set -eou pipefail

stage=1
stop_stage=2

ngpu=1

. ./utils/parse_options.sh

cmd="$cuda_cmd --gpu $ngpu"

eps=0.01
exp_dir=exp/adv-fgsm-$eps
mkdir -p $exp_dir
if [ $stage -le 1 ]; then
  #CUDA_VISIBLE_DEVICES=$(free-gpu -n $ngpu) python3 ./mmi_att_transformer_train_adv.py \
  $cmd $exp_dir/train.log ./mmi_att_transformer_train_adv.py \
  		      --world-size $ngpu \
  		      --raw true \
		      --full-libri false \
  		      --num-epochs 20 \
                      --max-duration 120 \
		      --adv fgsm \
  		      --fgsm-eps $eps \
  		      --exp $exp_dir
fi

if [ $stage -le 2 ] && [ $stop_stage -gt 2 ]; then
  #CUDA_VISIBLE_DEVICES=$(free-gpu) python3 ./mmi_att_transformer_decode_adv.py \
  $cmd $exp_dir/decode.log ./mmi_att_transformer_decode_adv.py \
		      --epoch 7 \
		      --avg 1 \
		      --max-duration 100 \
		      --exp $exp_dir \
		      --dataset test
fi
