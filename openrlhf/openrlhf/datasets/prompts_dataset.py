from torch.utils.data import Dataset
from tqdm import tqdm
import os

def apply_qwen_box_template_vision(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>" + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_qwen_tag_template_vision(question: str):
    return (
        "<|im_start|>system\nYou should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.<|im_end|>\n<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>" + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_r1_template_vision(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        "<|vision_start|><|image_pad|><|vision_end|>" + question
        + "\nAssistant: <think>"
    )

def apply_qwen_math_box_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_qwen_tag_template(question: str):
    return (
        "<|im_start|>system\nYou should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_llama3_box_template(question: str):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        + question
        + "\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )

def apply_no_template(question: str):
    return question


TEMPLATE_FACTORY = {
    "qwen_box_vision": apply_qwen_box_template_vision,
    "qwen_tag_vison": apply_qwen_tag_template_vision,
    "r1_vision": apply_r1_template_vision,
    "qwen_box_text": apply_qwen_math_box_template,
    "qwen_tag_text": apply_qwen_tag_template,
    "r1_text": apply_r1_template,
    "llama_text": apply_llama3_box_template,
    "no": apply_no_template,
}


def preprocess_data(data, input_template=None, input_key="input", label_key=None, media_key=None, media_dir=None, apply_chat_template=None) -> str:
    if media_dir is not None:
        media_path = os.path.join(media_dir,data[media_key][0])
    prompt = data[input_key]
    if apply_chat_template:
        if isinstance(prompt, str):
            if media_key is not None:
                prompt = [{"role": "user", "content": [{"type":"image","image":media_path},{"type":"text","text":prompt}]}]
            else:
                prompt = [{"role": "user", "content": prompt}]
        prompt = apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    elif input_template:
        prompt = input_template(prompt)
    else:
        prompt = [{"role": "user", "content": [{"type":"image","image": media_path},{"type":"text","text": prompt}]}]

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        processor: processor for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        processor,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.processor = processor

        # chat_template
        self.input_template = TEMPLATE_FACTORY[input_template] if input_template is not None else None
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        media_key = getattr(self.strategy.args, "media_key", None)
        media_dir = getattr(self.strategy.args, "media_dir", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.processor.apply_chat_template

        self.prompts = []
        self.labels = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, self.input_template, input_key, label_key, media_key, media_dir, apply_chat_template)
            self.prompts.append(prompt)
            self.labels.append(label)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx], self.labels[idx]
