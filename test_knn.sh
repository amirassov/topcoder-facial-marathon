#!/usr/bin/env bash


IMG_PREFIX=${1}
SOLUTION=${2}

SOLUTION_TRAIN=/wdata/solution_train.csv
EMBEDDINGS_PATH=/wdata/test_embeddings
KNN_PATH=/code/weights/knn.bin
KNN_PREDICTIONS=/wdata/test_nmslib_predictions.npz

PYTHONPATH=${TOPCODER_ROOT} python ${TOPCODER_ROOT}/reid/annotation_converter.py \
    --annotation=${SOLUTION} \
    --output=${SOLUTION_TRAIN}

rm -rf ${EMBEDDINGS_PATH}
mkdir ${EMBEDDINGS_PATH}
PYTHONPATH=${TOPCODER_ROOT} python ${TOPCODER_ROOT}/reid/predict_embeddings.py \
    --embedder_path=/code/weights/model-r100-ii/model \
    --mtcnn_path=/code/weights/mtcnn-model \
    --root=${IMG_PREFIX} \
    --annotation_path=${SOLUTION_TRAIN} \
    --output_path=${EMBEDDINGS_PATH} \
    --n_jobs=60

PYTHONPATH=${TOPCODER_ROOT} python ${TOPCODER_ROOT}/reid/predict_nmslib.py \
    --embedding_path=${EMBEDDINGS_PATH}/*.npz \
    --knn_path=${KNN_PATH} \
    --output_path=${KNN_PREDICTIONS} \
    --n_jobs=60

PYTHONPATH=${TOPCODER_ROOT} python ${TOPCODER_ROOT}/reid/prepare_solution.py \
    --output=${SOLUTION} \
    --detector_solution=${SOLUTION_TRAIN} \
    --nmslib_predictions=${KNN_PREDICTIONS} \
    --threshold=0.47

PYTHONPATH=${TOPCODER_ROOT} python ${TOPCODER_ROOT}/reid/annotation_converter.py \
    --annotation=${SOLUTION} \
    --output=${SOLUTION} \
    --to_test
