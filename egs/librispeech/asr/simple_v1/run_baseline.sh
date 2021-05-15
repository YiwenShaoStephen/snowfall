#!/usr/bin/env bash

# on clsp grid
. ./utils/cmd.sh
. path.sh

set -eou pipefail

stage=2
stop_stage=3

ngpu=1

. ./utils/parse_options.sh

cmd="$cuda_cmd --gpu $ngpu"
model_type=conformer
exp_dir=exp/${model_type}-baseline
mkdir -p $exp_dir
if [ $stage -le 1 ] && [ $stop_stage -gt 1 ]; then
  #CUDA_VISIBLE_DEVICES=$(free-gpu -n $ngpu) python3 ./mmi_att_transformer_train_adv.py \
  $cmd $exp_dir/train.log ./mmi_att_transformer_train_adv.py \
                      --world-size $ngpu \
  		      --raw true \
		      --full-libri false \
		      --model-type $model_type \
  		      --num-epochs 20 \
                      --max-duration 120 \
  		      --exp $exp_dir
fi

if [ $stage -le 2 ] && [ $stop_stage -gt 2 ]; then
  #CUDA_VISIBLE_DEVICES=$(free-gpu) python3 ./mmi_att_transformer_decode_adv.py \
  $cmd $exp_dir/decode.log ./mmi_att_transformer_decode_adv.py \
		      --epoch 3 \
		      --avg 1 \
		      --model-type $model_type \
		      --max-duration 100 \
		      --exp $exp_dir \
		      --dataset test
fi
