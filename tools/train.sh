#!/bin/bash
CUDA_VISIBLE_DEVICES=1,0 ./dist_train.sh ../configs/fabric/cascade_rcnn_r50_fpn_50e_coco.py 2
