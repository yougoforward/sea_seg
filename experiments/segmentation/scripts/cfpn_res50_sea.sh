# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset sea \
    --model cfpn --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname cfpn_res50_sea

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset sea \
    --model cfpn --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/sea/cfpn/cfpn_res50_sea/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset sea \
    --model cfpn --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/sea/cfpn/cfpn_res50_sea/model_best.pth.tar --split val --mode testval --ms

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset sea \
    --model cfpn --aux --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/sea/cfpn/cfpn_res50_sea/model_best.pth.tar --split test --mode test \
    --save-folder experiments/segmentation/results/cfpn_res50_sea