#!/usr/bin/env bash

IMG_PREFIX=${1}

MMDETECTION_ANN_FILE=/wdata/mmdetection/training_fix.pickle
OPEN_SET_FACE_ANN_FILE=${TOPCODER_ROOT}/data/training_fix.csv
RESIZE_IMG_PREFIX=/wdata/training_resize

MODEL=cascade_rcnn_dconv_c3-c5_r50_fpn
CONFIG_FILE=${TOPCODER_ROOT}/configs/${MODEL}.py

echo "prepare pretrained weights"
PYTHONPATH=${TOPCODER_ROOT} MPLBACKEND=AGG CUDA_VISIBLE_DEVICES=0 \
python ${TOPCODER_ROOT}/src/prune.py \
    --weights=/code/weights/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth \
    --output=/wdata/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166_part.pth

echo "prepare data for training"
mkdir ${RESIZE_IMG_PREFIX}
mkdir /wdata/mmdetection
PYTHONPATH=${TOPCODER_ROOT} python ${TOPCODER_ROOT}/src/convert.py \
    --annotation=${OPEN_SET_FACE_ANN_FILE} \
    --root=${IMG_PREFIX} \
    --output_root=${RESIZE_IMG_PREFIX} \
    --output=${MMDETECTION_ANN_FILE}

echo "train"
PYTHONPATH=${TOPCODER_ROOT} MPLBACKEND=AGG CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 ${TOPCODER_ROOT}/src/train.py \
    ${CONFIG_FILE} \
    --launcher pytorch \
    --img_prefix=${RESIZE_IMG_PREFIX} \
    --ann_file=${MMDETECTION_ANN_FILE}

echo "copy best checkpoint to /topcoder-facial-marathon/weights"
cp /wdata/cascade_rcnn_dconv_c3-c5_r50_fpn/latest.pth /code/weights/my_best_checkpoint.pth
