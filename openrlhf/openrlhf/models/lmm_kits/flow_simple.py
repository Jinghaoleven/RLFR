import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, List
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn import functional as F
import deepspeed
from transformers.utils import cached_file
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
import os


class FlowReward(nn.Module):
    """Velocity Deviation"""
    def __init__(self, target_channels, cond_channels, depth, width, rewarding_timesteps="0.8", eps=1e-6, grad_checkpointing=False):
        super(FlowReward, self).__init__()
        
        model_fn = SimpleMLPAdaLNTimePos
        self.model = model_fn(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            cond_channels=cond_channels,
            num_res_blocks=depth,
            eps=eps,
            grad_checkpointing=grad_checkpointing,
        )
        self.inference_timesteps = [float(timestep) for timestep in rewarding_timesteps.split("+")]
        self.debias_weight = [timestep / (1 - timestep) for timestep in self.inference_timesteps]

    def forward(self, target, cond, layer=None):
        """
        shape: (bsz*seq_len*diff_mul, dim)
        """
        noise = torch.randn_like(target)

        timestep_losses = []
        for idx, t_step in enumerate(self.inference_timesteps):
            t = torch.full((len(target),), t_step, device=target.device) 

            # linear interpolation between target and noise
            xt, ut = expand_t(t, target) * target + (1 - expand_t(t, target)) * noise, target - noise
            layer_id = torch.ones(size=(len(target),), device=target.device) * layer
            
            model_output = self.model(xt, t, cond, layer_id)
            loss = (model_output.float() - ut.float()) ** 2
            loss = torch.mean(loss, dim=list(range(1, len(loss.size()))))  # Take the mean over all non-batch dimensions.
            timestep_losses.append(loss)
            
        timestep_scores = [self.debias_weight[idx] * loss for loss in timestep_losses]
        loss = sum(timestep_losses)/len(timestep_losses)
        score = sum(timestep_scores)/len(timestep_scores)
        return loss, score
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        state_dict: Optional[dict] = None,
        **kwargs,
    ):  
        if state_dict is None:
            state_dict = self.model.state_dict()

        save_path = os.path.join(save_directory,"flow_model.bin")
        torch.save(state_dict, save_path)


#################################################################
def modulate(x, shift, scale):
    if x.dim()==3 and shift.dim()==2:
        shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
    return x * (1 + scale) + shift

def expand_t(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
        t: [bsz,], time vector
        x: [bsz,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def randn_tensor(shape, noise_repeat, device, dtype=torch.float32):
    bsz = shape[0]
    if bsz % noise_repeat != 0:
        raise ValueError(f"Batch size ({bsz}) must be divisible by noise repeat ({noise_repeat})")
    _shape = (noise_repeat,) + shape[1:]
    _tensor = torch.randn(_shape, device=device, dtype=dtype).repeat(bsz // noise_repeat, 1)
    return _tensor


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)[None]
        self.register_buffer('freqs', freqs)

    # @staticmethod
    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        args = t[:,None].float() * self.freqs.to(t.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels,
        eps=1e-6
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = RMSNorm(channels, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        if x.dim()==3 and gate_mlp.dim()==2:
            gate_mlp = gate_mlp.unsqueeze(1)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels, eps=1e-6):
        super().__init__()
        self.norm_final = RMSNorm(model_channels, eps=eps)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



#################################################################
class SimpleMLPAdaLNTimePos(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param cond_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        cond_channels,
        num_res_blocks,
        eps,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.grad_checkpointing = grad_checkpointing

        self.input_proj = nn.Linear(in_channels, model_channels)
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(cond_channels, model_channels)
        self.layer_embed = TimestepEmbedder(model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                eps=eps
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.final_layer = FinalLayer(model_channels, out_channels, eps=eps)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c, l=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = x.to(c.dtype)
        x = self.input_proj(x)
        c = self.cond_embed(c)
        t = self.time_embed(t)
        l = self.layer_embed(l)

        y = l + c + t

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)
        
        output = self.final_layer(x, y)

        return output
