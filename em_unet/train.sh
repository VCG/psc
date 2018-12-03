#!/bin/bash

source /n/coxfs01/leek/anaconda/miniconda2/bin/activate root
source activate caffe

export MODEL_DIR=/n/coxfs01/fgonda/bmvc/em_unet/model/1_1
export BMVC_SRC=/n/coxfs01/fgonda/bmvc/em_unet/src
export PYTHONPATH=$BMVC_SRC/PyGreentea/:$C2D/libs/malis/:/n/coxfs01/leek/caffe2/caffe/python

export CUDA_VISIBLE_DEVICES=1

cd $MODEL_DIR
echo "$BMVC_SRC/unet.py"
python $BMVC_SRC/unet.py  --action train --output $MODEL_DIR --train_device 1 --test_device 1 --augment 1 --transform 1 --random_blacks 0 --data_path /n/coxfs01/fgonda/experiments/p2d/500-distal/input/original/500-distal/im_uint8.h5 --data_name main --seg_path /n/coxfs01/fgonda/experiments/p2d/500-distal/input/original/500-distal/groundtruth_seg_thick.h5 --seg_name main

# end of program
exit 0;
