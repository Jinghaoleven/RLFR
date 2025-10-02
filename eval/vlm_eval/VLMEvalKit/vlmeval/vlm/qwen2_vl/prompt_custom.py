from __future__ import annotations
import os
from vlmeval.dataset import DATASET_TYPE
from ...smp import *

def build_mcq_cot_prompt(line, prompt, dataset, default=False):
    if default:
        cot_prompt = (
                "Answer the preceding multiple choice question. The last line of your response should follow "
                "this format: 'Answer: \\boxed{$LETTER}' (without quotes), where LETTER is one of the options. "
                "If you are uncertain or the problem is too complex, make a reasoned guess based on the "
                "information provided. Avoid repeating steps indefinitely—provide your best guess even if "
                "unsure. Think step by step logically, considering all relevant information before answering."
            )
        prompt = prompt + '\n' + cot_prompt
    else:
        if listinstr(['hall'], dataset):
            cot_prompt = (
                "Answer the preceding multiple choice question. You first thinks about "
                "the reasoning process in the mind and then provides the user with the answer. The answer "
                "is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. "
                "The answer should be enclosed within <answer> </answer> tags, i.e., Since $1+1=2$, so the "
                "answer is $2$. <answer> The answer is $\\\\boxed{2}$ </answer>, which means the final answer "
                "assistant's output should start with <answer> and end with </answer>. Think step by step logically, "
                "considering all relevant information before answering."
            )
            prompt = prompt.replace("Please select the correct answer from the options above.", '').strip()
            prompt = prompt + '\n' + cot_prompt
        elif listinstr(['MMMU_DEV_VAL'], dataset):
            cot_prompt = (
                "Solve the question. The user asks a question, and you solves it. You first thinks about "
                "the reasoning process in the mind and then provides the user with the answer. The answer "
                "is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} "
                "command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
                "respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is "
                "$\\\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>."
            )
            prompt = prompt.replace("Please select the correct answer from the options above.", '').strip()
            prompt = cot_prompt + '\n' + prompt
        else:
            cot_prompt = (
                "Answer the preceding multiple choice question. The last line of your response should follow "
                "this format: 'Answer: \\boxed{$LETTER}' (without quotes), where LETTER is one of the options. "
                "If you are uncertain or the problem is too complex, make a reasoned guess based on the "
                "information provided. Avoid repeating steps indefinitely—provide your best guess even if "
                "unsure. Think step by step logically, considering all relevant information before answering."
            )
            prompt = prompt + '\n' + cot_prompt

    return prompt


def build_qa_cot_prompt(line, prompt, dataset, default=False):
    if default:
        cot_prompt = (
            "Answer the preceding question. The last line of your response should follow this format: "
            "'Answer: \\boxed{$FINAL_ANSWER}' (without quotes), where 'FINAL_ANSWER' is your conclusion "
            "based on the reasoning provided. If you are uncertain or the problem is too complex, make "
            "a reasoned guess based on the information provided. Avoid repeating steps indefinitely—"
            "provide your best guess even if unsure. Think step by step logically, considering all "
            "relevant information before answering."
        )
        prompt = prompt + '\n' + cot_prompt
    else:
        if listinstr(['MathVista'], dataset):
            cot_prompt = (
                "Solve the question. The user asks a question, and you solves it. You first thinks about "
                "the reasoning process in the mind and then provides the user with the answer. The answer "
                "is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} "
                "command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
                "respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is "
                "$\\\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>."
            )
            prompt = cot_prompt + '\n' + prompt
        
        elif listinstr(['MathVerse'], dataset):
            cot_prompt = (
                "Solve the question. The user asks a question, and you solves it. You first thinks about "
                "the reasoning process in the mind and then provides the user with the answer. The final answer "
                "should be wrapped using the \\\\boxed{} command and enclosed within <answer> </answer> tags, i.e., "
                "Since $1+1=2$, so the answer is $2$. <answer> $\\\\boxed{2}$ </answer>."
            )
            pos_prompt = "Think step by step logically, considering all relevant information before answering."
            prompt = cot_prompt + '\n' + prompt + '\n' + pos_prompt

        elif listinstr(['MathVision'], dataset):
            cot_prompt = (
                "Answer the preceding question. You first thinks about "
                "the reasoning process in the mind and then provides the user with the answer. The answer "
                "is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. "
                "The answer should be enclosed within <answer> </answer> tags, i.e., Since $1+1=2$, so the "
                "answer is $2$. <answer> The answer is $\\\\boxed{2}$ </answer>, which means the final answer "
                "assistant's output should start with <answer> and end with </answer>. Think step by step logically, "
                "considering all relevant information before answering."
            )
            prompt = prompt + '\n' + cot_prompt
        
        elif listinstr(['LogicVista'], dataset):
            cot_prompt = (
                "Answer the preceding question. The last line of your response should follow this format: "
                "'Answer: \\boxed{$FINAL_ANSWER}' (without quotes), where 'FINAL_ANSWER' is your conclusion "
                "based on the reasoning provided. If you are uncertain or the problem is too complex, make "
                "a reasoned guess based on the information provided. Avoid repeating steps indefinitely—"
                "provide your best guess even if unsure. Think step by step logically, considering all "
                "relevant information before answering."
            )
            prompt = prompt + '\n' + cot_prompt
        else:
            cot_prompt = (
                "Solve the question. The user asks a question, and you solves it. You first thinks about "
                "the reasoning process in the mind and then provides the user with the answer. The answer "
                "is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. "
                "The answer should be enclosed within <answer> </answer> tags, i.e., Since $1+1=2$, so the "
                "answer is $2$. <answer> The answer is $\\\\boxed{2}$ </answer>, which means the final answer "
                "assistant's output should start with <answer> and end with </answer>."
            )
            prompt = cot_prompt + '\n' + prompt 
    return prompt



class Qwen2VLPromptMixin:
    """
    Mixin class for Qwen2VLChat to build custom prompt for different datasets.

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(self, *args, use_custom_prompt: bool = True, use_cot: bool = False, default_prompt: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt
        self.use_cot = use_cot
        self.default_prompt = default_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return True
        if dataset_type == 'MCQ':
            if dataset is not None and 'LEGO' in dataset:
                return False
            return True
        if dataset_type == 'Y/N' and dataset in {'HallusionBench', 'POPE', 'MME', 'AMBER'}:  # MME has it's own prompt
            return True
        if dataset_type == 'VQA' and dataset not in {'MMVet'}:  # MMVet VQA has it's own prompt
            return True
        return False

    def build_prompt(self, line, dataset: str):
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = self._build_mcq_prompt(line, dataset)
            if self.use_cot and not listinstr(['WeMath'], dataset):
                prompt = build_mcq_cot_prompt(line, prompt, dataset, default=self.default_prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath', 'QSpatial',
                            'LogicVista'], dataset):
                prompt = question
                if self.use_cot:
                    prompt = build_qa_cot_prompt(line, prompt, dataset, default=self.default_prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if self.use_cot:
                prompt = build_qa_cot_prompt(line, prompt)
        
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MCQ dataset: use chinese prompt if the question contains chinese characters."""
        MCQ_CN_PROMPT = '\n请回答选项字母。'
        MCQ_EN_PROMPT = '\nPlease select the correct answer from the options above.'
        
        # MCQ_CN_PROMPT = "你应该首先思考大脑中的推理过程，然后再向用户提供答案。请从上方的选项中选择正确答案。推理过程和答案应分别包含在 <think> 和 </think> 以及 <answer> 和 </answer> 标签中，这意味着你的输出应以 <think> 开始，以 </answer> 结束。"
        # MCQ_EN_PROMPT = "\nYou should first thinks about the reasoning process in the mind and then provides the user with the answer. Please select the correct answer from the options above. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, which means your output should start with <think> and end with </answer>."

        import string
        import pandas as pd

        def cn_string(s):
            import re

            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            # question = hint + '\n' + question
            question = f'Hint: {hint}\n' + f'Question: {question}\n'

        options = {
            cand: line[cand] 
            for cand in string.ascii_uppercase 
            if cand in line and not pd.isna(line[cand])
        }
        prompt = question

        if len(options):
            prompt += 'Options:\n'
            for key, item in options.items():
                prompt += f'{key}. {item}\n'
        
        if not listinstr(['WeMath'], dataset):
            if len(options):
                prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
            else:
                prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'
        prompt = prompt.rstrip()
        return prompt
