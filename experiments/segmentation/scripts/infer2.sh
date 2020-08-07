# !/usr/bin/env bash
#test [single-scale]
python -m experiments.segmentation.infer2 --dataset sea \
    --model cfpn --base-size 513 --crop-size 513 \
    --backbone resnet50 --resume experiments/segmentation/runs/sea/cfpn/cfpn_res50_sea/model_best.pth.tar --split val --mode testval
