#!/bin/bash
rm -rf /var/lib/storage/resources/cv_fall_2022/lesson_6_nn_intro/experiments/sgd_30.m_1.d_1 ? \
    && mkdir /var/lib/storage/resources/cv_fall_2022/lesson_6_nn_intro/experiments/sgd_30.m_1.d_1 \
    && cd /home/david_tyuman/my_github/cv_fall_2022 \
    && source ~/.pyenv/versions/3.8.2/envs/mac_env/bin/activate \
    && python make_pool.py --config=./lesson_6_nn_intro/configs/pool_making/base.json \
    && python train.py --net=./lesson_6_nn_intro/configs/architectures/simple_cnn.json --learn=./lesson_6_nn_intro/configs/learning_process/sgd_30.m_1.d_1.json
