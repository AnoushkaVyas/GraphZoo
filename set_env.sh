#!/usr/bin/env bash
export GRAPHZOO_HOME=$(pwd)
export LOG_DIR="$GRAPHZOO_HOME/logs"
export PYTHONPATH="$GRAPHZOO_HOME:$PYTHONPATH"
export DATAPATH="$GRAPHZOO_HOME/data"
source activate GraphZoo  # replace with source GraphZoo/bin/activate if you used a virtualenv
