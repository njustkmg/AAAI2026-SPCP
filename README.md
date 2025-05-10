# SPCP: Shaping Parameter Contribution Patterns for Out-of-Distribution Detection

## Dataset Preparation

The experiment is conducted based on the authoritative benchmarks provided by [OpenOOD v1.5](https://github.com/Jingkang50/OpenOOD), which include a variety of ID, Near-OOD, and Far-OOD datasets for comprehensive evaluation:

> - ID: CIFAR-10
>      > Near-OOD: `CIFAR-100`, `TinyImageNet`;<br>
>      > Far-OOD: `MNIST`, `SVHN`, `Texture`, `Places365`;<br>
> - ID: CIFAR-100
>      > Near-OOD: `CIFAR-10`, `TinyImageNet`;<br>
>      > Far-OOD: `MNIST`, `SVHN`, `Texture`, `Places365`;<br>
> - ID: ImageNet-200
>      > Near-OOD: `SSB-hard`, `NINCO`;<br>
>      > Far-OOD: `iNaturalist`, `Texture`, `OpenImage-O`;<br>

To streamline the dataset preparation process, all required datasets can be automatically downloaded and organized by executing the following shell script:
```
sh ./scripts/download/download.sh
```

## Preliminaries
It is run under Ubuntu Linux 18.04 and Python 3.8.19 environment, and requires some packages to be installed.
* [PyTorch](https://pytorch.org/)
* [numpy](http://www.numpy.org/)

## Run

### 1. CIFAR-10 Benchmark

```
# SPCP Train
sh scripts/ood/spcp/cifar10_train_spcp.sh <GPU_ID>
# SPCP Test
sh cifar10_test.sh
```

### 2. CIFAR-100 Benchmark

```
# SPCP Train
sh scripts/ood/spcp/cifar100_train_spcp.sh <GPU_ID>
# SPCP Test
sh cifar100_test.sh
```

### 3. ImageNet-200 Benchmark

```
# SPCP Train
sh scripts/ood/spcp/imagenet200_train_spcp.sh <GPU_ID>
# SPCP+LogitNorm Train
sh scripts/ood/spcp/imagenet200_train_spcp_logitnorm.sh <GPU_ID>
# SPCP Test
sh imagenet200_test.sh
# SPCP+LogitNorm Test
sh imagenet200_logitnorm_test.sh
```







>>>>>>> e62fa02 (first commit)
