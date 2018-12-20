import os
import glob
import shutil
import argparse
import json
import numpy as np
import cv2

from config import global_config
CFG = global_config.cfg

# supporting:
import matplotlib.pyplot as plt


def init_args():
    '''
    Arguments
    '''

    parser = argparse.ArgumentParser()

    # Arguments for saving
    parser.add_argument('--train-root', type=str, default='/home/ubuntu/dev/LaneNet-Pytorch/data/training_data/', help='Training dataset path (absolute path)')
    parser.add_argument('--image-path', type=str, default='image/', help='Directory name for images (relative to training)')
    parser.add_argument('--instance-path', type=str, default='gt_image_instance/', help='Directory name of instance images (relative to training)')
    parser.add_argument('--binary-path', type=str, default='gt_image_binary/', help='Directory name of binary images (relative to training)')

    # Arguments for dataset source
    parser.add_argument('--dataset-path', type=str, default='/home/ubuntu/haruya_dataset/tuSimple/train_set/', help='root dir of the dataset that contains video clips and json for label')

    return parser.parse_args()


def delete_files(path):
    '''
    Remove files or subdirectories in a directory
    '''
    for f in glob.glob(path+'*'):
        try:
            if os.path.isfile(f):
                os.unlink(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
        except Exception as e:
            print(e)


def create_data_structure(args):
    '''
    Organize and create directory to put data in
    '''
    img_path = args.train_root + args.image_path
    instance_path = args.train_root + args.instance_path
    binary_path = args.train_root + args.binary_path
    dirs = [args.train_root, img_path, instance_path, binary_path]

    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

    return img_path, instance_path, binary_path


def get_gt_json(args):
    '''
    Return ground truth json files for creating binary and instance images
    '''
    json_files = [f for f in os.listdir(args.dataset_path) if f.endswith('json')]
    jsons = []
    for j in json_files:
        jsons.append([json.loads(line) for line in open(args.dataset_path + j)])
    return jsons


def create_images(original, lanes):
    '''
    Returns binary and instance images
    '''
    binary = np.zeros_like(original)
    binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    instance = binary.copy()

    num_lines = len(lanes)
    for i, lane in enumerate(lanes):
        # create binary image
        cv2.polylines(binary, np.int32([lane]), isClosed=False, color=255, thickness=5)
        # create instance image
        cv2.polylines(instance, np.int32([lane]), isClosed=False, color=(i+1)*255/num_lines, thickness=5)

    return binary, instance


if __name__ == '__main__':

    # init args
    args = init_args()


    # delete all files from these directories before starting
    #FIXME: may not need this function
    delete_files(args.train_root)

    # create directories to save training images
    create_data_structure(args)

    # list dates:
    jsons = get_gt_json(args)
    print('Total json files: ', len(jsons))

    # create a text file for saving directory
    text_filename = args.train_root + 'train.txt'

    num = 0  # counter
    for i in range(len(jsons)):
        # should yield total of 3626 images
        gt_json = jsons[i]

        for gt in gt_json:
            # read json data:
            raw_file = gt['raw_file']
            y_samples = gt['h_samples']
            lanes = gt['lanes']

            print(raw_file)

            # get source image
            img = cv2.imread(args.dataset_path + raw_file)

            # create ground truth lanes
            gt_lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in lanes]

            # create binary and instance images
            binary, instance = create_images(img, gt_lanes)

            # create save path for each image
            img_name = '0'*(4-len(str(num))) + str(num) + '.png'  # 4 digits 
            source_img = args.train_root + args.image_path + img_name
            binary_img = args.train_root + args.binary_path + img_name
            instance_img = args.train_root + args.instance_path + img_name

            # save images to path
            cv2.imwrite(source_img, img)
            cv2.imwrite(binary_img, binary)
            cv2.imwrite(instance_img, instance)

            # add to txt file:
            data_line = source_img + ' ' + binary_img + ' ' + instance_img + '\n'
            with open(text_filename, 'a') as f:
                f.write(data_line)

            num += 1

    print('Total images: ', num)
