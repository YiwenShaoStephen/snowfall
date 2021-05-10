#!/usr/bin/env bash
set -eou pipefail

stage=1

eps=0.01
iter=2
alpha=$(echo $eps $iter | awk '{ print $1 * 1.5 / $2}') # alpha = 1.5 * eps / iter
exp_dir=exp/adv-pgd-$iter-$eps-full
mkdir -p $exp_dir
if [ $stage -le 1 ]; then
  CUDA_VISIBLE_DEVICES=$(free-gpu) python3 ./mmi_att_transformer_train_adv.py \
  		      --world-size 1 \
  		      --raw true \
		      --full-libri true \
  		      --num-epochs 20 \
                      --max-duration 120 \
		      --adv pgd \
		      --pgd-iter $iter \
  		      --pgd-eps $eps \
		      --pgd-alpha $alpha \
  		      --exp $exp_dir | tee -a $exp_dir/log.txt
fi
#done
exit 0
if [ $stage -le 2 ]; then
  CUDA_VISIBLE_DEVICES=$(free-gpu) python3 ./mmi_att_transformer_decode_adv.py \
		      --epoch 19 \
		      --avg 1 \
		      --max-duration 80 \
		      --exp $exp_dir
fi
