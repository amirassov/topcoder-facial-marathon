FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update -y && apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    unzip \
    yasm \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    build-essential \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libavformat-dev \
    libpq-dev \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    software-properties-common \
    libturbojpeg

RUN add-apt-repository -y ppa:jonathonf/python-3.6 \
    && apt-get update -y \
    && apt-get install -y python3.6 python3.6-dev \
    && ln -sfn /usr/bin/python3.6 /usr/local/bin/python \
    && ln -sfn /usr/bin/python3.6 /usr/bin/python3 \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && rm get-pip.py \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --upgrade --no-cache-dir pip && pip install --no-cache-dir \
    cycler==0.10.0 \
    dill==0.2.8.2 \
    h5py==2.7.1 \
    imgaug==0.2.5 \
    matplotlib==2.2.2 \
    numpy==1.14.3 \
    opencv-contrib-python==3.4.2.17 \
    pandas==0.23.0 \
    Pillow==5.1.0 \
    scikit-image==0.13.1 \
    scikit-learn==0.19.1 \
    scipy==1.1.0 \
    setuptools==39.1.0 \
    six==1.11.0 \
    tqdm==4.23.4 \
    ipython==7.3.0 \
    ipdb==0.12 \
    ninja==1.9.0 \
    yacs==0.1.6 \
    albumentations==0.2.2 \
    click==7.0 \
    pytest-runner==4.4 \
    jpeg4py==0.1.4 \
    cython==0.29.6 \
    nmslib==1.7.3.6 \
    mxnet==1.3.1

RUN pip install --upgrade --no-cache-dir cython && pip install --no-cache-dir pycocotools==2.0.0 mmcv==0.2.5
RUN pip install --no-cache-dir torch==1.0.0 torchvision==0.2.2

COPY . /code

WORKDIR /code/weights
RUN wget https://www.dropbox.com/s/09xiyd4nukpsexz/my_best_checkpoint.pth
RUN wget https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip && \
    unzip model-r100-arcface-ms1m-refine-v2.zip
RUN wget https://www.dropbox.com/s/63t3lakuygyfqsl/mtcnn-model.zip && \
    unzip mtcnn-model.zip
RUN wget https://www.dropbox.com/s/rqn0v6kk9v3bst2/knn.bin
ADD https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth .

WORKDIR /code/mmdetection
RUN bash compile.sh && \
    python setup.py develop

WORKDIR /code
ENV TOPCODER_ROOT /code
