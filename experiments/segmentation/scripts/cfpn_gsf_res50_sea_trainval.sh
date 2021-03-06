# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset sea --train-split trainval \
    --model cfpn_gsf --aux --base-size 321 --crop-size 321 \
    --backbone resnet50 --checkname cfpn_gsf_res50_sea

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset sea \
    --model cfpn_gsf --aux --base-size 321 --crop-size 321 \
    --backbone resnet50 --resume experiments/segmentation/runs/sea/cfpn_gsf/cfpn_gsf_res50_sea/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset sea \
    --model cfpn_gsf --aux --base-size 321 --crop-size 321 \
    --backbone resnet50 --resume experiments/segmentation/runs/sea/cfpn_gsf/cfpn_gsf_res50_sea/model_best.pth.tar --split val --mode testval --ms

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset sea \
    --model cfpn_gsf --aux --base-size 321 --crop-size 321 \
    --backbone resnet50 --resume experiments/segmentation/runs/sea/cfpn_gsf/cfpn_gsf_res50_sea/model_best.pth.tar --split test --mode test \
    --save-folder experiments/segmentation/results/cfpn_gsf_res50_sea