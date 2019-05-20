#!/bin/bash

python CIFAR10-C_test.py \
  --batchsize 128 \
  --model_name 'NAT' \
  --model_path '/model_path/VGG_NAT.pkl' \
  --distotion_root '/distotion_root/'
