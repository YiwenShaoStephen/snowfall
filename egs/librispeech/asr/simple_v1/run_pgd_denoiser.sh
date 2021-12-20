#!/usr/bin/env bash

set -eou pipefail
. path.sh
stage=1

eps=0.01
iter=7
alpha=$(echo $eps $iter | awk '{ print $1 * 1.5 / $2}') # alpha = 1.5 * eps / iter
exp_dir=exp/adv-pgd-denoiser
mkdir -p $exp_dir
if [ $stage -le 1 ]; then
  CUDA_VISIBLE_DEVICES=$(free-gpu) python3 ./mmi_att_transformer_train_adv.py \
  		      --world-size 1 \
  		      --raw true \
		      --full-libri true \
		      --raw-audio true \
  		      --num-epochs 20 \
                      --max-duration 20 \
		      --adv pgd \
		      --pgd-iter $iter \
  		      --pgd-eps $eps \
		      --fine-tune-mdl /export/c23/pzelasko/exp-conformer-noam-mmi-att-musan-sa-vgg/epoch-21.pt \
		      --denoiser-model-ckpt /export/c06/skataria/models/BWE-1/111/47_0.37262.pt \
		      --exp $exp_dir | tee -a $exp_dir/log.txt
fi
#done
#exit 0
if [ $stage -le 7 ]; then
  CUDA_VISIBLE_DEVICES=$(free-gpu) python3 ./mmi_att_transformer_decode.py \
		      --epoch 19 \
		      --avg 1 \
		      --max-duration 80 \
		      --exp $exp_dir
fi
