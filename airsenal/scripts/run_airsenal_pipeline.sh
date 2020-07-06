#!/bin/bash

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --weeks_ahead)
    WEEKS_AHEAD="$2"
    shift # past argument
    shift # past value
    ;;
    --num_free_transfers)
    NUM_FREE_TRANSFERS="$2"
    shift # past argument
    shift # past value
    ;;
    --bank)
    BANK="$2"
    shift # past argument
    shift # past value
    ;;
    --num_thread)
    NUM_THREAD="$2"
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# Get number of available CPUs (Mac OSX)
NCPU=`sysctl -n hw.ncpu`

# Set defaults if arguments not passed
NUM_THREAD=${NUM_THREAD:-$NCPU}
WEEKS_AHEAD=${WEEKS_AHEAD:-3}
BANK=${BANK:-0}
NUM_FREE_TRANSFERS=${NUM_FREE_TRANSFERS:-1}

echo "NUM_THREAD         = ${NUM_THREAD}"
echo "WEEKS_AHEAD        = ${WEEKS_AHEAD}"
echo "BANK               = ${BANK}"
echo "NUM_FREE_TRANSFERS = ${NUM_FREE_TRANSFERS}"

# Activate airsenalenv environment
CONDA_BASE=`conda info --base`
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate airsenalenv

# Remove any pre-existing AIrsenal data
rm /tmp/data.db

# Run full AIrsenal pipeline
# with default arguments but on all cores
setup_airsenal_database
update_airsenal_database --noattr
run_airsenal_predictions --num_thread $NCPU --weeks_ahead $WEEKS_AHEAD
run_airsenal_optimization --num_thread $NCPU --weeks_ahead $WEEKS_AHEAD --bank $BANK --num_free_transfers $NUM_FREE_TRANSFERS
