from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.utils import cached_file
from flash_attn.utils.distributed import all_gather

from .ring_attn_utils import convert_ring_attn_params, set_hacked_position_ids, clear_hacked_position_ids
from .utils import log_probs_from_logits, reset_position_ids
from openrlhf.models.lmm_kits.utils import get_generation_cls
from .lmm_kits.flow_simple import FlowReward
from .lmm_kits.utils import load_config
from functools import partial


class Actor_Flow(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for selecting actions based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        pretrained_flow_model=None,
        flow_config=None,
        strategy=None,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.args = strategy.args
            #There is no AutoModelForConditionalGeneration in transformers. We manually implement it.
            config = AutoConfig.from_pretrained(pretrain_or_model)
            if self.args.mllm_training:
                model_cls = get_generation_cls(config)
            elif self.args.llm_training:
                model_cls = AutoModelForCausalLM
            
            self.model = model_cls.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            if hasattr(config, "hidden_size"):
                hidden_size = config.hidden_size
            if hasattr(config, "word_embed_proj_dim"):
                hidden_size = config.word_embed_proj_dim
            elif hasattr(config, "is_encoder_decoder"):
                if config.is_encoder_decoder and hasattr(config, "decoder"):
                    if hasattr(config.decoder, "hidden_size"):
                        hidden_size = config.decoder.hidden_size

            # Initialize matching module
            flow_config = load_config(flow_config)
            flow_args = {
                "target_channels": hidden_size,
                "cond_channels": hidden_size,
                "width": hidden_size,
                "depth": flow_config.get("num_layers_flow"),
                "rewarding_timesteps": self.args.rewarding_timesteps,
                "eps": config.rms_norm_eps if not hasattr(config, "text_config") else config.text_config.rms_norm_eps,
                "grad_checkpointing": True,
            }
            self.flow = FlowReward(**flow_args).to(self.model.dtype)

            # Load pretrained flow model
            if pretrained_flow_model is not None:
                hook_kwargs = {"path_or_repo_id": pretrained_flow_model, "cache_dir": pretrained_flow_model, "token": None}
                try:
                    hook_file = cached_file(filename="flow_model.bin", **hook_kwargs)
                    hook_state_dict = torch.load(hook_file, map_location="cpu")
                    self.flow.load_state_dict(hook_state_dict, strict=True)
                    strategy.print(f"Successfully load flow from ({pretrained_flow_model}).")
                except Exception as err:
                    err_text = str(err)
                    strategy.print(f"Provided path ({pretrained_flow_model}) does not contain matching weights: {err_text}.")

            # Prepare flow config
            hook_layers_percentile = flow_config.get("hook_layers_percentile", None)
            if hook_layers_percentile is not None:
                hook_layers_percentile = [float(percentile) for percentile in hook_layers_percentile.split("+")]
            
            # Add flow hook to hidden states
            num_hidden_layers = config.num_hidden_layers if not hasattr(config, "text_config") else config.text_config.num_hidden_layers
            self.hook_layers = []
            for idx, hook_layer_percentile in enumerate(hook_layers_percentile):
                hook_layer = round(hook_layer_percentile * num_hidden_layers)
                hook_layer = hook_layer-1 if hook_layer == num_hidden_layers else hook_layer
                self.hook_layers.append(hook_layer)
                self.model.model.layers[hook_layer].register_forward_hook(partial(self.forward_hook, idx=idx))


            self.flow_context_modes = flow_config.get("flow_context_mode").split("+")
            self.flow_shift = [int(shift) for shift in flow_config.get("flow_shift").split("+")]
            self.flow_losses, self.flow_scores = [], []
            self.mode_weight = 1/len(self.flow_context_modes)
            assert len(self.flow_context_modes) == len(self.flow_shift)
            self.num_hidden_layers = num_hidden_layers
            self.num_hook_layers = len(hook_layers_percentile)

            self.forward_hook_enabled = True

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": kwargs.get("num_beams", 1) > 1,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        self.forward_hook_enabled = False

        # Call generate
        sequences = self.model.generate(**generate_args)

        self.forward_hook_enabled = True

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        logps_allgather=False,
        packed_seq_lens: Optional[list[int]] = None,
        visual_inputs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if visual_inputs is None:
            visual_inputs = {}
        '''
        for k,v in visual_inputs.items():
            if v.dtype == torch.float32:
                visual_inputs[k] = v.to(self.model.get_input_embeddings().weight.dtype)
        '''
        if self.args.mllm_training:
            inputs_embeds = self.model.get_inputs_embeds(sequences, **visual_inputs)
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                #position_ids = attention_mask.long().cumsum(-1) - 1
                #position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = self.model.get_position_ids(sequences,attention_mask=attention_mask, **visual_inputs)
            else:
                # convert attention_mask to position_ids
                packed_position_ids = self.model.get_position_ids(sequences, **visual_inputs)
                if ring_attn_group is not None:
                    labels = sequences
                    sequences, attention_mask, hacked_position_ids, inputs_embeds, split_position_ids = convert_ring_attn_params(
                        sequences, attention_mask, packed_seq_lens, ring_attn_group, inputs_embeds, packed_position_ids
                    )
                    position_ids = self.model.offset_split_position_ids(split_position_ids, hacked_position_ids) # this is true position_ids
                    #position_ids is directly hacked into flash_attn_forward to distinguish between different sequences
                else:
                    hacked_position_ids = reset_position_ids(attention_mask)
                    position_ids = self.model.offset_split_position_ids(packed_position_ids, hacked_position_ids)

                set_hacked_position_ids(hacked_position_ids)
                # explicitly ignore attention_mask for packing_samples
                attention_mask = None
            output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, **visual_inputs)
        elif self.args.llm_training:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                if ring_attn_group is not None:
                    sequences, attention_mask, position_ids = convert_ring_attn_params(
                        sequences, attention_mask, packed_seq_lens, ring_attn_group
                    )
                else:
                    # reset the positions for packed samples
                    position_ids = reset_position_ids(attention_mask)
            output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
        clear_hacked_position_ids()
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)

        if num_actions is None:
            assert return_output
            return output

        if not self.packing_samples:
            log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])
            action_log_probs = log_probs[:, -num_actions:]
        else:
            if ring_attn_group is not None and logps_allgather:
                rank = dist.get_rank(ring_attn_group)
                ring_attn_size = dist.get_world_size(ring_attn_group)
                total_seq_len = labels.numel()
                local_seq_len = total_seq_len // ring_attn_size
                local_slice = slice(rank * local_seq_len + 1, (rank + 1) * local_seq_len + 1)
                local_label = labels[:, local_slice]
                if rank == ring_attn_size - 1:
                    # add a dummy label to the last logit
                    local_label = F.pad(local_label, (0, 1), value=0)
                local_per_token_logps = torch.gather(
                    output["logits"].log_softmax(-1), dim=2, index=local_label.unsqueeze(2)
                ).squeeze(2)
                per_token_logps = all_gather(local_per_token_logps, ring_attn_group).reshape((1, -1))
                log_probs = per_token_logps[:, :-1]
            else:
                log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += seq_len
            action_log_probs = torch.cat(action_log_probs, dim=1)
        
        # debug use
        if len(self.flow_losses)==self.num_hook_layers*2:
            self.flow_losses = self.flow_losses[self.num_hook_layers:]
        if len(self.flow_scores)==self.num_hook_layers*2:
            self.flow_scores = self.flow_scores[self.num_hook_layers:]
            
        action_flow_loss = (sum(self.flow_losses)/len(self.flow_losses))[:, -num_actions:]
        action_flow_score = (sum(self.flow_scores)/len(self.flow_scores))[:, -num_actions:]
        self.flow_losses.clear()
        self.flow_scores.clear()

        if return_output:
            return (action_log_probs, output, action_flow_loss, action_flow_score)
        else:
            return action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
    
    def forward_hook(self, module, input, output, idx):
        if not self.forward_hook_enabled:     
            return
        hidden_states = output[0]
        flow_loss, flow_score = 0, 0
        # flow_context_mode only correspond to flow_shift and reflect the behavior in single hook
        for shift_idx, flow_context_mode in zip(self.flow_shift, self.flow_context_modes):
            context_hidden_states, target_hidden_states, layer_idx = self.prepare_input_for_matching(hidden_states, flow_context_mode, shift_idx, idx)
            mode_loss, mode_score= self.flow_matching(ctx=context_hidden_states.detach(), target=target_hidden_states, layer=layer_idx, context_mode=flow_context_mode, shift=shift_idx)
            flow_loss += self.mode_weight * mode_loss
            flow_score += self.mode_weight * mode_score
        self.flow_losses.append(flow_loss)
        self.flow_scores.append(flow_score)
        return output
    
    def prepare_input_for_matching(self, hidden_states, flow_context_mode, shift, layer_idx):
        if flow_context_mode== "token_pre":
            context_hidden_states = hidden_states[:,shift:,:]
            target = hidden_states[:,:-shift,:]
            layer_idx = self.hook_layers[layer_idx] / self.num_hidden_layers
        elif flow_context_mode== "token_post":
            context_hidden_states = hidden_states[:,:-shift,:]
            target = hidden_states[:,shift:,:]
            layer_idx = self.hook_layers[layer_idx] / self.num_hidden_layers
        elif flow_context_mode == "identity":
            context_hidden_states = target = hidden_states
            layer_idx = self.hook_layers[layer_idx] / self.num_hidden_layers
        return context_hidden_states, target, layer_idx
    
    def flow_matching(self, ctx, target, layer=None, context_mode=None, shift=None):
        """
        cond: torch.Size([bs,seq,dim])
        target: torch.Size([bs,hw,dim])
        mask: torch.Size([bs,seq])
        """
        bsz, seq_len, _ = target.shape
        """(bsz*seq_len*diff_mul, dim)"""
        ctx = ctx.reshape(bsz * seq_len, -1)
        target = target.reshape(bsz * seq_len, -1)

        loss, score = self.flow(cond=ctx, target=target, layer=layer)
        loss, score = loss.reshape(bsz,seq_len), score.reshape(bsz,seq_len)

        if context_mode == "token_pre":
            loss = torch.cat([loss, torch.zeros([bsz,shift],dtype=loss.dtype,device=loss.device)], dim=1)
            score = torch.cat([score, torch.zeros([bsz,shift],dtype=score.dtype,device=score.device)], dim=1)
        elif context_mode == "token_post":
            loss = torch.cat([torch.zeros([bsz,shift],dtype=loss.dtype,device=loss.device), loss], dim=1)
            score = torch.cat([torch.zeros([bsz,shift],dtype=score.dtype,device=score.device), score], dim=1)
            
        return loss, score
