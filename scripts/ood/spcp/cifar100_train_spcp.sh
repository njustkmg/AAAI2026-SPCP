#!/bin/bash

# sh ./scripts/ood/my/cifar100_train_spcp.sh 3

GPUID=$1

for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPUID python main.py \
        --config configs/datasets/cifar100/cifar100.yml \
        configs/datasets/cifar100/cifar100_ood.yml \
        configs/networks/spcp_net.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/pipelines/train/train_spcp.yml \
        --trainer.trainer_args.p 0.995 \
        --trainer.trainer_args.alpha 0.999 \
        --trainer.trainer_args.T 1000 \
        --seed ${seed} 
done


