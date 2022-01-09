#!/usr/bin/env bash
export GRAPHZOO_HOME=$(pwd)
export LOG_DIR="$GRAPHZOO_HOME/logs"
export PYTHONPATH="$GRAPHZOO_HOME:$PYTHONPATH"
export DATAPATH="$GRAPHZOO_HOME/data"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
source activate GraphZoo  # replace with source GraphZoo/bin/activate if you used a virtualenv
