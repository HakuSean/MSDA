#!/usr/bin/env bash

DATASET=$1
OPTION=$2
GPU=$3
SPEC=$4

TOOLS=/home/alfa/Documents/msda/dann/build/install/bin
LOG_FILE=logs/${DATASET}_logs/${OPTION}_${SPEC}.log
MODEL=/home/alfa/Documents/msda/cocktail/caffes/pretrain/pretrain.caffemodel

mkdir -p logs/${DATASET}_logs
echo "logging to ${LOG_FILE}"

#${MPI_BIN_DIR}mpirun -np $N_GPU \
$TOOLS/caffe train --solver=${OPTION}/${DATASET}_solver.prototxt --weights=${MODEL} --gpu=${GPU} 2>&1 | tee ${LOG_FILE}
