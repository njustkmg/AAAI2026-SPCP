#!/bin/bash

# sh ./scripts/ood/my/imagenet200_train_spcp_logitnorm.sh 4

GPUID=$1

for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPUID python main.py \
        --config configs/datasets/imagenet200/imagenet200.yml \
        configs/datasets/imagenet200/imagenet200_ood.yml \
        configs/networks/spcp_net.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/pipelines/train/train_spcp_logitnorm.yml \
        --network.backbone.name resnet18_224x224 \
        --trainer.trainer_args.p 0.99 \
        --optimizer.num_epochs 90 \
        --trainer.trainer_args.alpha 0.999 \
        --trainer.trainer_args.T 1000 \
        --seed ${seed} 
done
