MODEL_PATH=result/Qwen2.5-Math-7B/RL/.../global_step200_hf
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0

python -m eval.lm_eval.evaluate_model \
--model_name $MODEL_PATH \
--dataset_name eval/lm_eval/evaluation_suite \
--temperature 0 \
--max_tokens 8192 \
--template qwen_math_box \
--n_samples 1 \
--eval_type pass@k 