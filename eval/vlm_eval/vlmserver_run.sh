API_URL=  #(optional)

eval_model() {
  local run_name=$1
  local ckpt=$2
  local working_dir=$3
  local datasets=$5

  echo $run_name
  echo $working_dir
  echo $DATASETS
  # Launch the evaluated model
  vllm serve --trust-remote-code $run_name/ckpt/global_step${ckpt}_hf \
    --max-model-len=8192 \
    --generation-config ./config \
    --limit-mm-per-prompt image=36,video=2 \
    --gpu-memory-utilization 0.9 \
    --port 23333 \
    -tp 1 &
  VLLM_PID=$!
  trap "kill $VLLM_PID" EXIT INT TERM
  sleep 180s
  # Run evaluation
  python eval/vlm_eval/VLMEvalKit/run.py \
        --data $DATASETS \
        --model Qwen2.5-VL-vllm \
        --work-dir $working_dir \
        --mode infer \
        --use_cot \
        --reuse \
        --verbose \
        --api-nproc 16  
  kill $VLLM_PID

  # Judge with Local Model
  lmdeploy serve api_server /inspire/hdd/global_user/zhangjinghao-240108110057/models/Qwen/Qwen2.5-7B-Instruct --server-port 23320 --session-len 16384 &
  LMDEPOLY_PID=$!
  trap "kill $LMDEPOLY_PID" EXIT INT TERM
  sleep 200s
  python eval/vlm_eval/VLMEvalKit/run_judge.py \
        --data $DATASETS \
        --model Qwen2.5-VL-vllm \
        --work-dir $working_dir \
        --mode all \
        --reuse \
        --judge-api-base http://0.0.0.0:23320/v1/chat/completions \
        --verbose \
        --api-nproc 16 
  # kill $LMDEPOLY_PID

  # Judge with API model
  # python eval/vlm_eval/VLMEvalKit/run_judge.py \
  #       --data $DATASETS \
  #       --model Qwen2.5-VL-vllm \
  #       --work-dir $working_dir \
  #       --mode all \
  #       --reuse \
  #       --judge deepseek \
  #       --judge-api-base $API_URL \
  #       --verbose \
  #       --api-nproc 16 
}


# # ###################### Project 2 ###############################
RUN_NAME=result/Qwen2_5VL-7B/RL/Qwen2.5-VL-7B-Instruct/RLFR
BASE_NAME=$(basename $RUN_NAME)
MODEL_TYPE=Qwen2.5-VL-7B-Instruct
CKPT=50
result_dir=rlcustom-8192
RESULT_DIR=eval_result/$MODEL_TYPE/$BASE_NAME/$CKPT-$result_dir
DATASETS="MathVista_MINI MathVision_MINI MathVerse_MINI WeMath LogicVista VisuLogic"
eval_model $RUN_NAME $CKPT $RESULT_DIR
