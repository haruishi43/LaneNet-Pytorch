# LaneNet (Pytorch Implementaiton)

An Pytorch implementation of for real time lane detection based on ["Towards End-to-End Lane Detection: an Instance Segmentation Approach"](https://arxiv.org/pdf/1802.05591.pdf) which was featured in CVPR workshop for being 3rd place in the [TuSimple Lane Detection competition](http://benchmark.tusimple.ai/#/).

Referenced [TensorFlow implementation of LaneNet](https://github.com/MaybeShewill-CV/lanenet-lane-detection).

## Requirements:

- Python >= 3.5
- [PyTorch](https://pytorch.org/)

To install all dependencies:

```
pip install -r requirements.txt
```

## Dataset:

### Download TuSimple:

~~Dataset is publicly available at [this repo](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection) by TuSimple.~~

__Edit__: Train, test and GTs are available from [this wiki](https://github.com/TuSimple/tusimple-benchmark/wiki) or run the script below:
```
./scripts/download_tusimple.sh
```

### Preprocessing:

Preprocessing script is used as below:
```
mkdir -p data/tusimple
python scripts/preprocess_tusimple_dataset.py --src-dir data/tusimple
```

__Note__: This script also could be used to download the dataset, but will not always work, so use `scripts/download_tusimple.sh` for downloading the dataset.

## Usage:

### Training:

```
python scripts/run_tusimple.py --run-type train --exp-config configs/default.yaml
```

### Evaluation:

```
python scripts/run_tusimple.py --run-type test --exp-config configs/default.yaml
```

## TODO:

- [ ] debug discriminative loss since mean IoU isn't going up during training


## Acknowledgements:

- [pytorch-lanenet](https://github.com/klintan/pytorch-lanenet)
- [lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection)
