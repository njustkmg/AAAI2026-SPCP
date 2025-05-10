#!/bin/bash
# sh imagenet200_logitnorm_test.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

# ood
CUDA_VISIBLE_DEVICES=4 python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root './results/imagenet200_spcp_net_spcp_logitnorm_e90_lr0.1_alpha0.999_p0.99_logitnorm_default_last' \
   --postprocessor msp \
   --save-score --save-csv


