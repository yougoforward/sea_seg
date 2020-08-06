# !/usr/bin/env bash

#test [single-scale]
python -m experiments.segmentation.test_single_image --datasets sea --model cfpn_gsf --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/sea/cfpn_gsf/cfpn_gsf_res50_sea/model_best.pth.tar \
    --input-path datasets/sea_707/image --save-path experiments/segmentation/sea_707