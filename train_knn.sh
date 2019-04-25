#!/usr/bin/env bash

IMG_PREFIX=${1}

EMBEDDINGS_PATH=/wdata/training_embeddings
KNN_PATH=/code/weights/knn.bin

rm -rf ${EMBEDDINGS_PATH}
mkdir ${EMBEDDINGS_PATH}
PYTHONPATH=${TOPCODER_ROOT} python ${TOPCODER_ROOT}/reid/predict_embeddings.py \
    --embedder_path=/code/weights/model-r100-ii/model \
    --mtcnn_path=/code/weights/mtcnn-model \
    --root=${IMG_PREFIX} \
    --annotation_path=${TOPCODER_ROOT}/data/training_fix_reid.csv \
    --output_path=${EMBEDDINGS_PATH} \
    --n_jobs=80

PYTHONPATH=${TOPCODER_ROOT} python ${TOPCODER_ROOT}/reid/fit_nmslib.py \
    --embedding_path=${EMBEDDINGS_PATH}/*.npz \
    --knn_path=${KNN_PATH} \
    --n_jobs=80
