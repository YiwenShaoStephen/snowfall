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

model_type=contextnet
eps=0.01
iter=7
alpha=$(echo $eps $iter | awk '{ print $1 * 1.5 / $2}') # alpha = 1.5 * eps / iter
exp_dir=exp/${model_type}-adv-pgd-finetune-$iter-$eps
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
		      --adv pgd \
		      --pgd-iter $iter \
  		      --pgd-eps $eps \
		      --pgd-alpha $alpha \
		      --start-epoch 5 \
		      --finetune-dir exp/baseline-context \
		      --exp $exp_dir | tee -a $exp_dir/log.txt
fi

if [ $stage -le 2 ] && [ $stop_stage -gt 2 ]; then
  #CUDA_VISIBLE_DEVICES=$(free-gpu) python3 ./mmi_att_transformer_decode_adv.py \
  $cmd $exp_dir/decode.log ./mmi_att_transformer_decode_adv.py \
                      --model-type $model_type \
		      --epoch 2 \
		      --avg 1 \
		      --max-duration 100 \
		      --exp $exp_dir \
		      --dataset test
fi
