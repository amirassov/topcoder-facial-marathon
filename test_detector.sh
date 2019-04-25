#!/usr/bin/env bash


IMG_PREFIX=${1}
SOLUTION=${2}

MMDETECTION_ANN_FILE=/wdata/mmdetection/test.pickle
OPEN_SET_FACE_ANN_FILE=/wdata/test.csv
MODEL=cascade_rcnn_dconv_c3-c5_r50_fpn
CONFIG_FILE=${TOPCODER_ROOT}/configs/${MODEL}.py
PREDICTION=/wdata/${MODEL}/test.pkl
CHECKPOINT=/code/weights/my_best_checkpoint.pth

PYTHONPATH=${TOPCODER_ROOT} \
python ${TOPCODER_ROOT}/src/prepare_test.py \
    --output=${OPEN_SET_FACE_ANN_FILE} \
    --root=${IMG_PREFIX}

mkdir /wdata/mmdetection
PYTHONPATH=${TOPCODER_ROOT} \
python ${TOPCODER_ROOT}/src/convert.py \
    --annotation=${OPEN_SET_FACE_ANN_FILE} \
    --root=${IMG_PREFIX} \
    --output=${MMDETECTION_ANN_FILE}

PYTHONPATH=${TOPCODER_ROOT} MPLBACKEND=AGG CUDA_VISIBLE_DEVICES=0 \
python ${TOPCODER_ROOT}/src/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    --gpus 1 \
    --proc_per_gpu=1 \
    --out=${PREDICTION} \
    --ann_file=${MMDETECTION_ANN_FILE} \
    --img_prefix=${IMG_PREFIX}

PYTHONPATH=${TOPCODER_ROOT} MPLBACKEND=AGG \
python ${TOPCODER_ROOT}/src/prepare_solution.py \
    --annotation=${MMDETECTION_ANN_FILE} \
    --predictions=${PREDICTION} \
    --output=${SOLUTION}
