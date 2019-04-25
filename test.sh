#!/usr/bin/env bash

IMG_PREFIX=${1}
SOLUTION=${2}

bash test_detector.sh ${IMG_PREFIX} ${SOLUTION}
bash test_knn.sh ${IMG_PREFIX} ${SOLUTION}