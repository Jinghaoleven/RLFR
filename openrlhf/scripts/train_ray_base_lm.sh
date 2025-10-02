#!/bin/bash
# =================== User Configuration ===================
# Please modify these variables according to your environment
# =========================================================

# Base paths - MODIFY THESE
WORKSPACE_DIR=./openrlhf  # Path to project root directory

# Experiment Setting
DATASET_PATH=./dataset/rltrain/math_lvl3to5_8k   # Path to your dataset
PRETRAIN_MODEL_PATH=Qwen/Qwen2.5-Math-7B   # Path to pretrained model
SAVE_PATH=./result/Qwen2_5-Math-7B/RL        # Path to save checkpoints
LOG_PATH=./log_dir
export NODE_RANK=0
export MASTER_ADDR="127.0.0.1"

# Model configuration
PROJECT_NAME="Qwen2.5-Math-7B-math_lvl3to5_8k"
EXP_NAME="Base-GRPO"              # Name for this training run
RUN_NAME=$PROJECT_NAME/$EXP_NAME
cd $WORKSPACE_DIR
# =================== Script Execution ===================
# You shouldn't need to modify anything below this line
# ======================================================

# Get script PID and setup directories
SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR=$LOG_PATH/$PROJECT_NAME             # tensorboard log
export CUR_LOG_DIR=$LOG_DIR/$EXP_NAME/$TIMESTAMP   # local log
export REWARD_LOG_PATH=$LOG_DIR/$EXP_NAME/reward.log


# start rm
if [ "$NODE_RANK" -eq 0 ]; then
    python -m openrlhf.cli.serve_rm \
        --mode rule \
        --tokenizer_path $PRETRAIN_MODEL_PATH \
        --max_gen_len 3072 \
        --data_path $DATASET_PATH \
        --input_key problem \
        --label_key answer \
        --template_type qwen \
        --port 5000 \
        --host $MASTER_ADDR &
fi

# Stop any existing ray processes
ray stop

# Create necessary directories
mkdir -p $SAVE_PATH/$RUN_NAME
mkdir -p $LOG_DIR
mkdir -p $CUR_LOG_DIR

# Print help information
echo "================================================================"
echo "LMM-R1 Direct RL Geometry Training"
echo "================================================================"
echo "Model name: $RUN_NAME"
echo "Dataset: $DATASET_PATH"
echo "Pretrained model: $PRETRAIN_MODEL_PATH"
echo "Logs will be saved to: $CUR_LOG_DIR"
echo
echo "To monitor logs:"
echo "  tail -f $CUR_LOG_DIR/train.log"
echo
echo "================================================================"

# Start ray
echo "Starting ray..."
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4 --temp-dir ~/.cache/ray

# Start training
echo "Starting training..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\"working_dir\": \"$WORKSPACE_DIR\"}" \
   -- python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --remote_rm_url http://$MASTER_ADDR:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.3 \
   --deepspeed_enable_sleep \
   --enable_prefix_caching \
   --pretrain $PRETRAIN_MODEL_PATH \
   --save_path $SAVE_PATH/$RUN_NAME \
   --micro_train_batch_size 1 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --temperature 1.0 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 5 \
   --max_samples 100000 \
   --prompt_max_len 3000 \
   --generate_max_len 1024 \
   --advantage_estimator group_norm \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --init_kl_coef 0.001 \
   --lambd 1 \
   --gamma 1 \
   --llm_training \
   --prompt_data $DATASET_PATH \
   --input_key problem \
   --label_key answer \
   --input_template qwen_box_text \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --vllm_sync_backend nccl \
   --gradient_checkpointing \
   --gradient_checkpointing_use_reentrant \
   --save_steps 50 \
   --ckpt_path $SAVE_PATH/$RUN_NAME/ckpt \
   --save_hf_ckpt \
   --load_checkpoint \
   --wandb_run_name $EXP_NAME \
   --use_tensorboard $LOG_DIR > >(tee -a "$CUR_LOG_DIR/train.log") 2>&1
TRAIN_PID=$!

# Record process IDs
echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"

# Wait for training to complete
echo "Training is running in the background. Check logs at ${CUR_LOG_DIR}/train.log"
echo "To attach to the training process: wait $TRAIN_PID"

# Uncomment to wait for training to complete before exiting
# wait $TRAIN_PID

# Cleanup instructions
echo "When finished, clean up with:"
echo "pkill -f openrlhf"
echo "ray stop"
echo "All logs are available in ${CUR_LOG_DIR}"