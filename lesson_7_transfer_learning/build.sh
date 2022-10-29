set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip \
        python3-tk \
        git

    pip3 install -qq \
        pytest \
        scikit-image \
        scikit-learn \
        opencv-python \
        matplotlib \
        imgaug \
        pandas \
        pytorch-ignite \
        albumentations==1.3.0 \
        torch==1.12.1 \
        torchvision==0.13.1 \
        pytorch_lightning==1.7.7 \
        efficientnet_pytorch \
        albumentations_experimental \
        timm==0.6.11 \
        moviepy

    pip3 install -qq git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
}

setup_checker() {
    python3 -c 'import matplotlib.pyplot'
}

"$@"