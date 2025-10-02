from __future__ import annotations

import os
import warnings
from PIL import Image

from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.vlm.qwen2_vl.prompt_custom import Qwen2VLPromptMixin

SYSTEM_PROMPT_7B = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. Th answer should be enclosed within <answer> </answer> tags, i.e., Since $1+1=2$, so the answer is $2$. <answer> The answer is $\\\\boxed{2}$ </answer>, which means the final answer assistant's output should start with <answer> and end with </answer>."

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


class Qwen2VLAPIvllm(Qwen2VLPromptMixin, BaseAPI):
    is_api: bool = True

    def __init__(
        self,
        api_base: str = None,
        key: str = 'sk-123456',
        timeout: int = 60,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_length=16384,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        seed=3407,
        use_custom_prompt: bool = True,
        use_cot: bool = False,
        default_prompt: bool = False,
        **kwargs,
    ):

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )

        self.timeout = timeout
        key = os.environ.get('LMDEPLOY_API_KEY', key)
        api_base = os.environ.get('LMDEPLOY_API_BASE', api_base)
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)'
        )
        self.key = key
        self.api_base = api_base
        super().__init__(use_custom_prompt=use_custom_prompt, use_cot=use_cot, default_prompt=default_prompt, **kwargs)

        # self.system_prompt = SYSTEM_PROMPT_7B
        model_url = ''.join([api_base.split('v1')[0], 'v1/models'])
        resp = requests.get(model_url)
        self.model = resp.json()['data'][0]['id']
        self.logger.info(f'vllm deploy model: {self.model}')
    
    def prepare_image(self, message):
        img = Image.open(message['value'])
        b64 = encode_image_to_base64(img)
        extra_args = message.copy()
        extra_args.pop('type')
        extra_args.pop('value')
        img_struct = dict(url=f'data:image/jpeg;base64,{b64}', **extra_args)
        return img_struct

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image_url', 'image_url': self.prepare_image(s)}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, inputs, **kwargs) -> str:

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append(
            {'role': 'user', 'content': self._prepare_content(inputs, dataset=kwargs.get('dataset', None))}
        )
        # if self.verbose:
        #     print(f'\033[31m{messages}\033[0m')

        # generate
        generation_kwargs = self.generate_kwargs.copy()
        kwargs.pop('dataset', None)
        kwargs.pop('model_path', None)
        kwargs.update(generation_kwargs)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=messages,
            n=1,
            **kwargs)
        
        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.8)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = 'Failed to obtain answer via API. '
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except:
            pass
        return ret_code, answer, response
