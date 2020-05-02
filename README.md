# LaneNet (Pytorch Implementaiton)

An Pytorch implementation of for real time lane detection based on ["Towards End-to-End Lane Detection: an Instance Segmentation Approach"](https://arxiv.org/pdf/1802.05591.pdf) which was featured in CVPR workshop for being 3rd place in the [TuSimple Lane Detection competition](http://benchmark.tusimple.ai/#/).

Referenced [TensorFlow implementation of LaneNet](https://github.com/MaybeShewill-CV/lanenet-lane-detection).


## Dataset:

~~Dataset is publicly available at [this repo](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection) by TuSimple.~~
Edit: Train, test and GTs are available from [this issue](https://github.com/TuSimple/tusimple-benchmark/issues/3) or run the script below:
```
./scripts/download_tusimple.sh
```

## Requirements:

- Python >= 3.5
- [PyTorch](https://pytorch.org/)

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

Create a directory to save the model:

```
mkdir data/saved_model
```

```
python train.py  --dataset-file /path/to/dataset/train.txt
```



### Testing:


## TODO:
