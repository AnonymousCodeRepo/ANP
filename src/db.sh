#!/bin/bash

python Decision_Boundary_Measurement.py \
  --batchsize 128 \
  --vectors_path '/vectors_path/OrthogonalVectors.npy' \
  --model_path '/model_path/VGG_NAT.pkl' \
  --dataset_path '/dataset_path/' \
  --margin_list_path '/margin_list_path/VGG_margin_list.pkl'
