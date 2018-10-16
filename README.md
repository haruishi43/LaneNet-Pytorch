# LaneNet (Pytorch Implementaiton)

An Pytorch implementation of for real time lane detection based on ["Towards End-to-End Lane Detection: an Instance Segmentation Approach"](https://arxiv.org/pdf/1802.05591.pdf) which was featured in CVPR workshop for being 3rd place in the [TuSimple Lane Detection competition](http://benchmark.tusimple.ai/#/).

Referenced [TensorFlow implementation of LaneNet](https://github.com/MaybeShewill-CV/lanenet-lane-detection).


## Dataset:

Dataset is publicly available at [this repo](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection) by TuSimple.

## Requirements:

- Python3
- [PyTorch](https://pytorch.org/)
- OpenCV

To install all dependencies:

```
pip install -r requirements.txt
```

## Usage:

### Creating the dataset:

- Since the tuSimple dataset does not provide binary and instance images, I provided `data_creater.py` script to create the needed dataset.
- Make sure to download the tuSimple dataset from the aforementioned link.

```
# create directories
mkdir data/training_data
mkdir data/training_data/gt_image_instance
mkdir data/training_data/gt_image_binary

python data_creater.py --dataset-path /path/to/tuSimple/dataset
```

### Training:

All configurations and hyperparameters for training is in `config/global_config.py`.

```
python train.py  --dataset-file /path/to/dataset/train.txt
```

### Testing:

WIP



## TODO:

- [ ] Debug
- [ ] CPU-only
- [ ] Visualization
- [ ] TensorboardX
- [ ] 