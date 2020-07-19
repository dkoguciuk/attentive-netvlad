#!/bin/bash
for it in $(seq 1 1 19)
do
        python3 evaluate.py --log_dir log_101 --ordering 'flat,norm,squeeze=256,cg,norm' --ckpt log_101/model_after_epoch_$it.ckpt --result_filename "after_epoch_$it.txt"
done
