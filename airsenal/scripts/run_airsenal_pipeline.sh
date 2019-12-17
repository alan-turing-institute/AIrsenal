#!/bin/bash

# Get number of available CPUs (Mac OSX)
NCPU=`sysctl -n hw.ncpu`

# Activate airsenalenv environment
CONDA_BASE=`conda info --base`
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate airsenalenv

# Remove any pre-existing AIrsenal data
rm /tmp/data.db

# Run full AIrsenal pipeline
# with default arguments but on all cores
setup_airsenal_database
update_airsenal_database
run_airsenal_predictions --num_thread $NCPU
run_airsenal_optimization --num_thread $NCPU
