#!/bin/bash
CUDA_VISIBLE_DEVICES=1,0 ./dist_test.sh ../configs/fabric/faster_rcnn_r50_fpn_2x_coco.py ../model_data/faster_rcnn_r50_fpn_2x_coco/epoch_50.pth 2 --eval bbox --out ../results/faster_rcnn_r50_fpn_2x_coco/result.pkl
