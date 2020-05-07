#!/usr/bin/env python3

import os
import os.path as osp
import argparse

from torch.utils.data import DataLoader

from LaneNet import (
    build_optimizer,
    build_lr_scheduler,
    get_config,
)
from LaneNet.datasets import tuSimpleDataset as Dataset
from LaneNet.losses import compute_loss
from LaneNet.models import LaneNet
from LaneNet.engines import Engine


def save_model(save_path, epoch, model):
    save_name = os.path.join(save_path, f'{epoch}_checkpoint.pth')
    torch.save(model, save_name)
    print("model is saved: {}".format(save_name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modifty config options from command line",
    )
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config
    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    Returns:
        None.
    """
    config = get_config(exp_config, opts)

    train_dataset_file = osp.join(
        config.DATASET.root,
        config.DATASET.train_txt,
    )
    val_dataset_file = osp.join(
        config.DATASET.root,
        config.DATASET.val_txt,
    )
    test_dataset_file = osp.join(
        config.DATASET.root,
        config.DATASET.test_txt,
    )
    if osp.exists(val_dataset_file):
        Dataset.create_val(
            train_dataset_file,
            val_dataset_file,
            config.DATASET.val_ratio)

    train_dataset = Dataset(
        dataset_path=train_dataset_file,
        mode='train',
        height=config.DATASET.height,
        width=config.DATASET.width,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DATASET.batch_size,
        shuffle=config.DATASET.shuffle,
        sampler=None,
        batch_sampler=None,
        num_workers=config.DATASET.num_workers,
        drop_last=True,
    )
    val_dataset = Dataset(
        dataset_path=val_dataset_file,
        mode='test',
        height=config.DATASET.height,
        width=config.DATASET.width,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DATASET.batch_size,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=config.DATASET.num_workers,
        drop_last=False,
    )

    model = LaneNet()
    model.cuda()

    optimizer = build_optimizer(
        model,
        optim=config.TRAIN.optim,
        lr=config.TRAIN.lr,
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        lr_scheduler=config.TRAIN.lr_scheduler,
        stepsize=config.TRAIN.step_size,
        gamma=config.TRAIN.gamma,
        max_epoch=config.TRAIN.max_epoch
    )
    print(f"{config.TRAIN.max_epoch} epochs {len(train_dataset)} training samples\n")

    # Init engine
    engine = Engine(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        use_gpu=config.use_gpu,
    )

    # Run engine
    engine.run(
        save_dir=config.save_dir,
        max_epoch=config.TRAIN.max_epoch,
        start_epoch=0,
        print_freq=config.TRAIN.print_freq,
        start_eval=config.TRAIN.start_eval,
        eval_freq=config.TRAIN.eval_freq,
    )

    test_dataset = Dataset(
        dataset_path=test_dataset_file,
        mode='test',
        height=config.DATASET.height,
        width=config.DATASET.width,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        drop_last=False,
    )

    engine.test(
        test_loader
    )


if __name__ == '__main__':
    main()