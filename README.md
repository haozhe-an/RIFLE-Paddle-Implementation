# RIFLE-Paddle-Implementation
This repository contains code for the implementation of RIFLE using [Paddlepaddle](https://www.paddlepaddle.org.cn/).

> [ICML'20] Xingjian Li*, Haoyi Xiong*, Haozhe An, Dejing Dou, and Cheng-Zhong Xu. RIFLE: Backpropagation in Depth for Deep Transfer Learning through Re-Initializing the Fully-connected LayEr. International Conference on Machine Learning (ICML’20), Vienna, Austria, 2020.

## Setup
- Download transfer learning target datasets, like [CUB_200_2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) or others. Arrange the dataset in this way:

```
    root/train/dog/xxy.jpg
    root/train/dog/xxz.jpg
    ...
    root/train/cat/nsdf3.jpg
    root/train/cat/asd932_.jpg
    ...

    root/test/dog/xxx.jpg
    ...
    root/test/cat/123.jpg
    ...
```

- Download [the pretrained models](https://github.com/PaddlePaddle/models/tree/release/1.7/PaddleCV/image_classification#resnet-series). We report the results of ResNet-50 below.

## Running Scripts

Modify `global_data_path` in `as_data_reader/data_path.py` to the path root where the dataset is.


Below are the commands to run baseline and RIFLE on Stanford Cars respectively,
```bash
python run.py --dataset Stanford_Cars --num_epoch 40 --batch_size 32 --fc_reinit 0 
python run.py --dataset Stanford_Cars --num_epoch 40 --batch_size 32 --fc_reinit 1 --cyclic_num 2 
```
Modify the argument appropriately to adjust the dataset folder name and the number of re-initializations of FC layer as desired.

## Results
We test cyclic_num = [2,3,4] and report the best results for RIFLE among the three. In this version of implementation, we avoid any re-initialization of FC layer when the training enters its last 5 epochs, so as to prevent the final model from being underfitted due to the lack of training budget after the last re-initialization.

Dataset | l2 | RIFLE
---|---|---
CUB_200_2011|0.7487|0.7618 (cyclic_num=3)
Stanford-Cars|0.8492|0.8610 (cyclic_num=4)
