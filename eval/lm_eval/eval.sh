MODEL_PATH=/inspire/hdd/global_user/zhangjinghao-240108110057/models
export VLLM_WORKER_MULTIPROC_METHOD=spawn


python ./evaluate_model.py \
--model_name $MODEL_PATH \
--dataset_name ./evaluation_suite \
--temperature 0 \
--max_tokens 8192 \
--template qwen_math_box \
--n_samples 1 \
--eval_type pass@k 