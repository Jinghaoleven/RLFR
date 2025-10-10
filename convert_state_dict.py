import os
import torch
import argparse

def replace_bin_state_dict(ckpt_dir):
    bin_path = os.path.join(ckpt_dir,"pytorch_model.bin")
    state_dict = torch.load(bin_path, map_location=torch.device('cpu'))
    save_path = bin_path.replace("pytorch_model.bin","flow_model.bin")

    # 创建一个新的字典，用于存储修改后的键
    flow_state_dict = {}
    # 修改键名称
    for key in state_dict:
        if key.startswith("flow"):
            new_key = key.replace("flow.", "")  # 示例：替换前缀
            flow_state_dict[new_key] = state_dict[key]
    
    torch.save(flow_state_dict, save_path)
    print(f"Successfully translate to {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default=None)
    args = parser.parse_args()

    replace_bin_state_dict(args.ckpt_dir)