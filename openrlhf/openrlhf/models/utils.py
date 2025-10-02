from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    if kl_estimator == "k1":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask

    # The k2 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # The k2_loss is approximately equivalent to the
    # one-step KL divergence penalty with the k1 estimator
    # used in https://arxiv.org/pdf/2310.10505.
    if kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    if kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        if action_mask is not None:
            log_ratio = log_ratio * action_mask
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    return log_ratio


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    flow_coef: float = None,
    action_score: Union[torch.Tensor, list[torch.Tensor]] = None,
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    if action_mask is not None:
        kl_reward = -kl_coef * kl

        action_flow_reward = 0
        if flow_coef !=0:
            action_flow_reward = -flow_coef * action_score

        eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
        last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

        reward = last_reward + kl_reward + action_flow_reward
    else:
        # TODO: write a more efficient version
        reward = []
        for i, (kl_seg, action_len) in enumerate(zip(kl, num_actions)):
            kl_reward = -kl_coef * kl_seg
            kl_reward[action_len - 1] += r[i]
            reward.append(kl_reward)

    return reward

def postprocess_flow_score(action_score, args, return_indices=False):

    if args.flow_score_threshold > 0:
        if args.flow_score_bil:
            action_score = torch.where((action_score>args.flow_score_threshold) | (action_score<-args.flow_score_threshold), action_score, 0)
        else:
            action_score = torch.where(action_score>args.flow_score_threshold, action_score, 0)

    if args.flow_score_topk > 0:
        batch_id_need_topk = torch.count_nonzero(action_score, dim=1) >= args.flow_score_topk
        reward_indices = None
        if batch_id_need_topk.any() and action_score.shape[1] >= args.flow_score_topk:
            
            if args.flow_score_bil:
                top_k_side = int(args.flow_score_topk // 2)
                values_pos, indices_pos = torch.topk(action_score, top_k_side, dim=1)
                values_neg, indices_neg = torch.topk(-action_score, top_k_side, dim=1)
                values = torch.cat([values_pos, -values_neg],dim=1)
                indices = torch.cat([indices_pos, indices_neg],dim=1)
                reward_indices = (indices_pos, indices_neg)
            else:
                values, indices = torch.topk(action_score, args.flow_score_topk, dim=1)
                reward_indices = (indices,)

            action_score_topk = torch.zeros_like(action_score).scatter_(dim=1, index=indices, src=values.to(action_score.dtype))
            action_score[batch_id_need_topk] = action_score_topk[batch_id_need_topk]
    
    if return_indices:
        return action_score, reward_indices
    return action_score


def normalize_flow_score(action_score, action_mask, args):
    action_score = action_score * action_mask
    if args.flow_reward_norm == "minmax":
        min_values = get_row_min(action_score)
        max_values, _ = torch.max(action_score, 1, keepdim=True)
        action_score = torch.where(action_mask,(action_score - min_values) / (max_values - min_values), action_score)
        if args.flow_score_bil:
            action_score[action_mask] = (action_score[action_mask] - 0.5) * 2
    elif args.flow_reward_norm == "mean":
        action_score = torch.where(action_mask,(action_score - min_values) / (max_values - min_values), action_score)
        sum_values, valid_cnt = (action_score * action_mask).sum(dim=1), action_mask.sum(dim=1)
        mean_values = sum_values / valid_cnt
        action_score = torch.where(action_mask, action_score - mean_values.unsqueeze(1),action_score)

    if args.flow_gamma!=1:
        action_score[action_mask] = action_score[action_mask].pow(args.flow_gamma)
    return action_score


def calculate_nonzero_percentage(tensor: torch.Tensor) -> float:
    # 统计0的数量
    zero_count = (tensor == 0).sum().float()
    # 计算总元素数量
    total_count = tensor.numel()
    # 计算百分比
    percentage = ((total_count - zero_count) / total_count)
    return percentage.item()

def get_row_min(x: torch.Tensor):

    # 找到每一行的最小值，形状保持 (n, 1) 方便广播
    row_min, _ = torch.min(x, dim=1, keepdim=True)

    # 把等于该行最小值的位置替换成 +∞
    inf = torch.tensor(float('inf'), dtype=x.dtype, device=x.device)
    x_masked = x.masked_fill(x == row_min, inf)

    # 再取每行最小值 -> 行的第二小唯一值
    second_min_row, _ = torch.min(x_masked, dim=1, keepdim=True)
    return second_min_row


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack(
            [torch.logsumexp(l, dim=-1) for l in logits]  # loop to reduce peak mem consumption
        )
        log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy

def tail_mask(mask):
    B, L = mask.shape

    # 找到每行 True 的起止位置
    idx_expand = torch.arange(L,device=mask.device).expand_as(mask)
    row_any = mask.any(dim=1)

    # 对于有 True 的行，计算 start 和 end
    start = torch.where(mask, idx_expand, L).min(dim=1).values
    end   = torch.where(mask, idx_expand, -1).max(dim=1).values + 1
    length = (end - start).clamp(min=1)  # 防止除零

    # cutoff：保留最后25%
    cutoff = start + (length * 3 // 4)

    # 构造 mask：只有在 [cutoff, end) 区间内才 True
    # idx_expand = idx.unsqueeze(0).expand(B, -1)
    tail_mask = (idx_expand >= cutoff.unsqueeze(1)) & (idx_expand < end.unsqueeze(1))

    # 更新 x
    mask = mask & tail_mask
    return mask


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    if mask.sum()==0:
        return (tensor * mask).sum(axis=dim)
    else:
        return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()


# Reset positions for packed samples
# For example
# Input: attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
# Output: position_ids  = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def unpacking_samples(values: torch.Tensor, packed_seqlens: list[int]):
    values = values.squeeze(0)
    unpacked_values = []
    offset = 0
    for seqlen in packed_seqlens:
        unpacked_values.append(values[offset : offset + seqlen])
        offset += seqlen
    return unpacked_values
