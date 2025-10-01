# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, is_torch_npu_available, is_torch_xpu_available, PreTrainedModel
from typing import Optional

from trl.models.modeling_base import PreTrainedModelWrapper
from transformers import PreTrainedModel
from functools import partial

from .flow_simple import FlowReward


class AutoModelForCausalLMWithFlowHook(PreTrainedModelWrapper):
    r"""
    An autoregressive model with a value head in addition to the language model head.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, `push_to_hub` and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the `ValueHead` class. Currently, the supported args are:
            - **summary_dropout_prob** (`float`, `optional`, defaults to `None`) -- The dropout probability for the
                `ValueHead` class.
            - **v_head_initializer_range** (`float`, `optional`, defaults to `0.2`) -- The initializer range for the
                `ValueHead` if a specific initialization strategy is selected.
            - **v_head_init_strategy** (`str`, `optional`, defaults to `None`) -- The initialization strategy for the
                `ValueHead`. Currently, the supported strategies are:
                - **`None`** -- Initializes the weights of the `ValueHead` with a random distribution. This is the default
                    strategy.
                - **"normal"** -- Initializes the weights of the `ValueHead` with a normal distribution.
    """

    transformers_parent_class = AutoModelForCausalLM
    supported_modules = ("flow",)
    supported_args = (
        "num_layers_flow",
        "hook_layers_percentile",
        "flow_batch_mul",
        "flow_context_mode",
        "flow_shift",
    )

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__(pretrained_model, **kwargs)
        flow_kwargs, _, _ = self._split_kwargs(kwargs)
        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        config = self.pretrained_model.config
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.flow = FlowReward(
            target_channels=hidden_size,
            cond_channels=hidden_size,
            width=hidden_size,
            depth=flow_kwargs["num_layers_flow"],
            eps=config.rms_norm_eps if not hasattr(config, "text_config") else config.text_config.rms_norm_eps,
            grad_checkpointing=True,
        ).to(self.pretrained_model.dtype)
        
        num_hidden_layers = config.num_hidden_layers if not hasattr(config, "text_config") else config.text_config.num_hidden_layers

        # Add flow hook to hidden states
        hook_layers_percentile = flow_kwargs["hook_layers_percentile"]
        self.hook_layers = []
        for idx, hook_layer_percentile in enumerate(hook_layers_percentile):
            hook_layer = round(hook_layer_percentile * num_hidden_layers)
            hook_layer = hook_layer-1 if hook_layer == num_hidden_layers else hook_layer
            self.hook_layers.append(hook_layer)
            self.pretrained_model.model.layers[hook_layer].register_forward_hook(partial(self.forward_hook, idx=idx))
        

        self.flow_batch_mul = flow_kwargs["flow_batch_mul"]
        self.flow_context_modes = flow_kwargs["flow_context_mode"]
        self.flow_shift = flow_kwargs["flow_shift"]
        self.flow_losses = []
        self.context_weight = 1/len(self.flow_context_modes)
        assert len(self.flow_context_modes) == len(self.flow_shift)
        self.num_hidden_layers = num_hidden_layers

    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the value head. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
        self.flow.model.initialize_weights()

    def prepare_input_for_matching(self, hidden_states, mask, flow_context_mode, shift, layer_idx):
        if flow_context_mode== "token_pre":
            response_hidden_states = hidden_states[:,shift:,:]
            target = hidden_states[:,:-shift,:]
            mask = mask[:,:-shift]
            layer_idx = self.hook_layers[layer_idx] / self.num_hidden_layers
        elif flow_context_mode== "token_post":
            response_hidden_states = hidden_states[:,:-shift,:]
            target = hidden_states[:,shift:,:]
            mask = mask[:,shift:]
            layer_idx = self.hook_layers[layer_idx] / self.num_hidden_layers
        elif flow_context_mode == "identity":
            response_hidden_states = hidden_states
            target = hidden_states
            mask = mask
            layer_idx = self.hook_layers[layer_idx] / self.num_hidden_layers
        return response_hidden_states, target, mask, layer_idx
        
    
    def forward_hook(self, module, input, output, idx):
        hidden_states = output[0]
        flow_loss = 0
        # flow_context_mode correspond to flow_shift and reflect the behavior in single hook_post
        for shift_idx, flow_context_mode in zip(self.flow_shift, self.flow_context_modes):
            response_hidden_states, target, mask, layer_idx = self.prepare_input_for_matching(hidden_states, self.mask, flow_context_mode, shift_idx, idx)
            flow_loss += self.context_weight * self.flow_matching(cxt=response_hidden_states, target=target.detach(), mask=mask, layer=layer_idx)
        self.flow_losses.append(flow_loss)
        return output

    def flow_matching(self, cxt, target, mask=None, layer=None):
        """
        z: torch.Size([bs,seq,dim])
        target: torch.Size([bs,hw,dim])
        mask: torch.Size([bs,seq])
        """
        bsz, seq_len, _ = target.shape
        """(bsz*seq_len*diff_mul, dim)"""
        cxt = cxt.reshape(bsz * seq_len, -1).repeat(self.flow_batch_mul, 1)
        target = target.reshape(bsz * seq_len, -1).repeat(self.flow_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.flow_batch_mul) if mask is not None else None

        loss = self.flow(cond=cxt, target=target, mask=mask, layer=layer)
        return loss

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            return_past_key_values (bool): A flag indicating if the computed hidden-states should be returned.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values
        kwargs.pop("num_items_in_batch")

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        self.mask = kwargs["labels"]!=-100
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        flow_losses = sum(self.flow_losses)/len(self.flow_losses)
        self.flow_losses.clear()

        metrics = {}
        metrics["losses/ntp"] = base_model_output.loss.detach().cpu().item()
        metrics["losses/matching"] = flow_losses.detach().cpu().item()

        return (flow_losses, base_model_output.logits, metrics)

    def generate(self, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model.
        Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
        method of the wrapped model for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the v_head
            pretrained_model_state_dict = {}

        # matching_state_dict = self.matching.model.state_dict(*args, **kwargs)
        # for k, v in diff_state_dict.items():
        #     pretrained_model_state_dict[f"matching.model.{k}"] = v
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]
            if isinstance(first_device, int):
                if is_torch_npu_available():
                    first_device = f"npu:{first_device}"
                elif is_torch_xpu_available():
                    first_device = f"xpu:{first_device}"
                else:
                    first_device = f"cuda:{first_device}"
            self.flow.model = self.flow.model.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True
