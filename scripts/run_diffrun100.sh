#!/bin/bash
set -ex

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
export TPU_HOST=${TPU_HOST:-10.255.128.3}
export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800
export TPU_NAME="${TPU_NAME:-tpu-v2-256-euw4a-21}"

export RUN_NAME="diffrun100c"
export GIN_CONFIG="configs/diffrun100.gin"
export BUCKET="dota-euw4a"
export DATASET_FILES="datasets/danbooru2019-s/danbooru2019-s-0*"
export MODEL_DIR="${MODEL_DIR:-runs}"

cores="$(echo $TPU_NAME | sed 's/^tpu-v[23][-]\([0-9]*\).*$/\1/g')"
if [ -z "$cores" ]
then
  1>&2 echo "Failed to parse TPU core count from $TPU_NAME"
  exit 1
fi
if echo $TPU_NAME | grep '[-]v3[-]'
then
  export BATCH_PER="${BATCH_PER:-8}" # tpu-v3
else
  export BATCH_PER="${BATCH_PER:-4}" # tpu-v2
fi
export BATCH_SIZE="${BATCH_SIZE:-$(($BATCH_PER * $cores))}"
export NUM_HOSTS="${NUM_HOSTS:-$(($cores / 8))}"

date="$(python3 -c 'import datetime; print(datetime.datetime.now().strftime("%Y-%m-%d"))')"
logfile="logs/${RUN_NAME}-${date}.txt"
mkdir -p logs

tmux-set-title "${RUN_NAME} ${TPU_NAME}"

export PYTHONPATH=.

while true; do
  timeout --signal=SIGKILL 19h python3 wrapper.py scripts/run_tfork.py train --bucket_name_prefix "${BUCKET}" --tpu_name "${TPU_NAME}" --exp_name "${RUN_NAME}" --log_dir "${MODEL_DIR}" --tfr_file "${DATASET_FILES}" --total_bs "${BATCH_SIZE}" --num_hosts "${NUM_HOSTS}" 2>&1 | tee -a "$logfile"
  if [ ! -z "$TPU_NO_RECREATE" ]
  then
    echo "Not recreating TPU."
    sleep 30
  else
    echo "Recreating TPU in 120s."
    sleep 120
    # sudo pip3 install -U tpudiepie
    pu recreate "$TPU_NAME" --yes
  fi
done
  
