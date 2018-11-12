#!/usr/bin/env sh

DATASET=$1 # dw2a, pacs2a
OPTION=$2 # baseline_alexnet
SPEC=$3 # the specialty, mainly for logs
GPU=$4 # nothing = 0 or 1,2

TOOLS=/home/alfa/Documents/msda/mywork/caffe/build/install/bin
LOG_FILE=logs/${DATASET}_logs/${OPTION}_${SPEC}.log
MODEL=/home/alfa/Documents/msda/mywork/pretrain/resnet-18.caffemodel

if [[ ${GPU} == "" ]]; then
    GPU=0
fi

mkdir -p logs/${DATASET}_logs
echo "logging to ${LOG_FILE}"

$TOOLS/caffe train --solver=models/${DATASET}/${OPTION}/solver.prototxt --weights=${MODEL} --gpu ${GPU} 2>&1 | tee ${LOG_FILE}
