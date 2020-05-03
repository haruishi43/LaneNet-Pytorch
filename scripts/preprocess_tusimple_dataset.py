#!/usr/bin/env python3

import argparse
import glob
import json
import os
import os.path as osp
import shutil
from tqdm import tqdm

import cv2
import numpy as np

TRAIN_URL = (
    "https://s3.us-east-2.amazonaws.com/"
    "benchmark-frontend/datasets/1/train_set.zip")
TEST_URL = (
    "https://s3.us-east-2.amazonaws.com/"
    "benchmark-frontend/datasets/1/test_set.zip")
TEST_LABEL_URL = (
    "https://s3.us-east-2.amazonaws.com/"
    "benchmark-frontend/truth/1/test_label.json")


def init_args():
    r"""
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src-dir',
        dest='src_dir',
        type=str,
        help='The origin path of unzipped tusimple dataset'
    )
    return parser.parse_args()


def download_tusimple(
    src_dir: str,
    train_set_path: str,
    test_set_path: str,
) -> None:
    r"""
    Download tuSimple datast
    :param src_dir:
    :param train_set_path:
    :param test_set_path
    """
    from subprocess import Popen, PIPE
    def run_cmd(cmd: list) -> None:
        p = Popen(cmd, stderr=PIPE, stdout=PIPE)
        ret, err = p.communicate()
        assert p.returncode == 0, f'{err.decode()}'
    
    run_cmd(['wget', '-P', src_dir, TEST_LABEL_URL])
    run_cmd(['wget', '-P', src_dir, TRAIN_URL])
    run_cmd(['wget', '-P', src_dir, TEST_URL])
    run_cmd(['unzip', '-d', train_set_path, src_dir+'/train_set.zip'])
    run_cmd(['unzip', '-d', test_set_path, src_dir+'/test_set.zip'])
    run_cmd(['rm', src_dir+'/train_set.zip'])
    run_cmd(['rm', src_dir+'/test_set.zip'])
    run_cmd(['mv', src_dir+'/test_label.json', test_set_path+'/'])
    print('Download complete!')


def process_json_file(
    json_file_path: str,
    src_dir: str,
    ori_dst_dir: str,
    binary_dst_dir: str,
    instance_dst_dir: str,
) -> None:
    r"""
    :param json_file_path:
    :param src_dir: origin clip file path
    :param ori_dst_dir:
    :param binary_dst_dir:
    :param instance_dst_dir:
    :return:
    """
    assert osp.exists(json_file_path), \
        f'{json_file_path} not exist'

    image_nums = len(os.listdir(ori_dst_dir))

    print(f'Preprocessing images')
    with open(json_file_path, 'r') as file:
        for line_index, line in enumerate(tqdm(file)):
            info_dict = json.loads(line)

            image_dir = osp.split(info_dict['raw_file'])[0]
            image_dir_split = image_dir.split('/')[1:]
            image_dir_split.append(osp.split(info_dict['raw_file'])[1])
            image_name = '_'.join(image_dir_split)
            image_path = osp.join(src_dir, info_dict['raw_file'])
            assert osp.exists(image_path), '{:s} not exist'.format(image_path)

            h_samples = info_dict['h_samples']
            lanes = info_dict['lanes']

            image_name_new = '{:s}.png'.format('{:d}'.format(line_index + image_nums).zfill(4))

            src_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            dst_binary_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)
            dst_instance_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8)

            for lane_index, lane in enumerate(lanes):
                assert len(h_samples) == len(lane)
                lane_x = []
                lane_y = []
                for index in range(len(lane)):
                    if lane[index] == -2:
                        continue
                    else:
                        ptx = lane[index]
                        pty = h_samples[index]
                        lane_x.append(ptx)
                        lane_y.append(pty)
                if not lane_x:
                    continue
                lane_pts = np.vstack((lane_x, lane_y)).transpose()
                lane_pts = np.array([lane_pts], np.int64)

                cv2.polylines(
                    dst_binary_image, lane_pts, isClosed=False,
                    color=255, thickness=5)
                cv2.polylines(
                    dst_instance_image, lane_pts, isClosed=False,
                    color=lane_index * 50 + 20, thickness=5)

            dst_binary_image_path = osp.join(binary_dst_dir, image_name_new)
            dst_instance_image_path = osp.join(instance_dst_dir, image_name_new)
            dst_rgb_image_path = osp.join(ori_dst_dir, image_name_new)

            cv2.imwrite(dst_binary_image_path, dst_binary_image)
            cv2.imwrite(dst_instance_image_path, dst_instance_image)
            cv2.imwrite(dst_rgb_image_path, src_image)

            # print('Process {:s} success'.format(image_name))


def gen_sample(
    dst_file: str,
    b_gt_image_dir: str,
    i_gt_image_dir: str,
    image_dir: str,
) -> None:
    r"""
    generate sample index file
    :param src_dir:
    :param b_gt_image_dir:
    :param i_gt_image_dir:
    :param image_dir:
    :return:
    """
    print(f'Creating txt file @ {dst_file}')
    with open(dst_file, 'w') as file:
        for image_name in tqdm(os.listdir(b_gt_image_dir)):
            if not image_name.endswith('.png'):
                continue

            binary_gt_image_path = osp.join(b_gt_image_dir, image_name)
            instance_gt_image_path = osp.join(i_gt_image_dir, image_name)
            image_path = osp.join(image_dir, image_name)

            assert osp.exists(image_path), f'{image_path} not exist'
            assert osp.exists(instance_gt_image_path), f'{instance_gt_image_path} not exist'

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None or i_gt_image is None:
                continue
            else:
                info = '{:s} {:s} {:s}\n'.format(image_path, binary_gt_image_path, instance_gt_image_path)
                file.write(info)


def process_tusimple_dataset(src_dir: str) -> None:
    r"""
    :param src_dir:
    :return:
    """
    assert osp.exists(src_dir), f"ERR: {src_dir} does not exist!"

    # Sources
    train_src_path = osp.join(src_dir, 'train_set')
    test_src_path = osp.join(src_dir, 'test_set')

    if not (osp.exists(train_src_path) and osp.exists(test_src_path)):
        # Download from URL
        print('Downloading tuSimple dataset')
        download_tusimple(src_dir, train_src_path, test_src_path)

    # Destinations
    train_dst_path = osp.join(src_dir, 'train')
    test_dst_path = osp.join(src_dir, 'test')
    os.makedirs(train_dst_path, exist_ok=True)
    os.makedirs(test_dst_path, exist_ok=True)

    for json_label_path in glob.glob(f'{train_src_path}/label*.json'):
        json_label_name = osp.split(json_label_path)[-1]
        shutil.copyfile(json_label_path, osp.join(train_dst_path, json_label_name))

    for json_label_path in glob.glob(f'{test_src_path}/test_label.json'):
        json_label_name = osp.split(json_label_path)[-1]
        shutil.copyfile(json_label_path, osp.join(test_dst_path, json_label_name))

    # Training Dataset
    gt_image_dir = osp.join(train_dst_path, 'gt_image')
    gt_binary_dir = osp.join(train_dst_path, 'gt_binary_image')
    gt_instance_dir = osp.join(train_dst_path, 'gt_instance_image')

    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)

    for json_label_path in glob.glob(f'{train_dst_path}/*.json'):
        process_json_file(json_label_path, train_src_path, gt_image_dir, gt_binary_dir, gt_instance_dir)

    dst_file = osp.join(train_dst_path, 'train.txt')
    gen_sample(dst_file, gt_binary_dir, gt_instance_dir, gt_image_dir)

    # Testing Dataset
    gt_image_dir = osp.join(test_dst_path, 'gt_image')
    gt_binary_dir = osp.join(test_dst_path, 'gt_binary_image')
    gt_instance_dir = osp.join(test_dst_path, 'gt_instance_image')

    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)

    for json_label_path in glob.glob(f'{test_dst_path}/*.json'):
        process_json_file(json_label_path, test_src_path, gt_image_dir, gt_binary_dir, gt_instance_dir)

    dst_file = osp.join(test_dst_path, 'test.txt')
    gen_sample(dst_file, gt_binary_dir, gt_instance_dir, gt_image_dir)


if __name__ == '__main__':
    args = init_args()
    process_tusimple_dataset(args.src_dir)