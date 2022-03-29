#!/bin/bash

base_url=https://mecmodels.s3-us-west-1.amazonaws.com

for model in UGRNN_relu UGRNN_relu_reward CueUGRNN_relu
do
    mkdir -p ./mecmodels/${model}/ckpts
    curl -fLo ./mecmodels/${model}/ckpts/ckpt-101.data-00000-of-00002 ${base_url}/${model}/ckpt-101.data-00000-of-00002
    curl -fLo ./mecmodels/${model}/ckpts/ckpt-101.data-00001-of-00002 ${base_url}/${model}/ckpt-101.data-00001-of-00002
    curl -fLo ./mecmodels/${model}/ckpts/ckpt-101.index ${base_url}/${model}/ckpt-101.index
done