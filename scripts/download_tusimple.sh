#!/bin/bash
#
# Download tusimple dataset to `data` directory
#
mkdir -p data/tusimple
cd data/tusimple
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip
unzip -d train_set train_set.zip
unzip -d test_set test_set.zip
# wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_baseline.json
wget -P test_set https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json
rm train_set.zip
rm test_set.zip