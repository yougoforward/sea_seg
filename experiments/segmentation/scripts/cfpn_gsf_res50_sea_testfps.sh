# !/usr/bin/env bash
#test [single-scale]
python -m experiments.segmentation.test_fps_params --model cfpn_gsf --backbone resnet50 --crop-size 321