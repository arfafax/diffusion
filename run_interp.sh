#######
# Generate progressive samples and output them as an MP4 video.
# Usage: OUTPUT_BUCKET=dota-euw4a run_samples.sh diffrun101
#######

# The TPU to run sampling on
TPU_NAME=tpu-v3-8-euw4a-203

# The bucket and path where the checkpoint is
MODEL_BUCKET=${MODEL_BUCKET:-dota-euw4a}
MODEL_PATH=${MODEL_PATH:-runs}

# The bucket to output the samples to
OUTPUT_BUCKET=${OUTPUT_BUCKET:-tensorfork-arfa-euw4}

# The run name
RUN_NAME=${1:-e621-s-2}

# Get the latest checkpoint step
CHECKPOINT=$(gsutil-latest-checkpoint gs://$MODEL_BUCKET/$MODEL_PATH/$RUN_NAME | sed -rn 's/.*-([0-9]+)/\1/p')
echo $CHECKPOINT

# Generate samples and output them to gs://$OUTPUT_BUCKET/ddpm-samples/$RUN_NAME
PYTHONPATH=. python3 scripts/run_tfork_samples.py simple_eval --mode interp --tpu_name $TPU_NAME --load_ckpt model.ckpt-$CHECKPOINT --model_dir gs://$MODEL_BUCKET/$MODEL_PATH/$RUN_NAME --samples_dir gs://$OUTPUT_BUCKET/ddpm-samples/$RUN_NAME/ --bucket_name_prefix $MODEL_BUCKET --tfr_file datasets/octo2k/octo2k-0* --dump_samples_only True --once True --num_hosts 1

# Get the samples and make them into a video. The file will be named $RUN_NAME_step$CHECKPOINT.mp4
#./make_video.sh gs://$OUTPUT_BUCKET/ddpm-samples/$RUN_NAME/samples_xstartpred_ema1_step000${CHECKPOINT}_part000000.pkl ${RUN_NAME}_step${CHECKPOINT}
