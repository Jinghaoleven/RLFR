export CUDA_VISIBLE_DEVICES=0,1,2,3

llamafactory-cli train examples/train_flow/qwen2_5vl_7b_flow.yaml
llamafactory-cli train LLaMA-Factory/examples/train_flow/qwen2_5_7b_flow.yaml