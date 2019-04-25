#!/usr/bin/env bash

IMG_PREFIX=${1}

bash train_detector.sh ${IMG_PREFIX}
bash train_knn.sh ${IMG_PREFIX}
