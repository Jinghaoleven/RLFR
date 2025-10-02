import os
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, masked_mean, unpacking_samples, entropy_from_logits, calculate_nonzero_percentage, normalize_flow_score, postprocess_flow_score
from openrlhf.utils.distributed_sampler import DistributedSampler

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer


class PPOTrainer(ABC):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        tokenizer (Callable, optional): Tokenizer for input data.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        reward_fn (Callable, optional): Custom reward function for computing rewards.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        flow_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        flow_scheduler = None,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        data_processor: Optional[Callable[[Any], Dict]] = None,
        data_tokenizer: Optional[Callable[[Any], Dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.data_processor = data_processor
        self.tokenizer = data_processor.tokenizer if self.args.mllm_training else data_tokenizer
        self.processor = data_processor.processor if self.args.mllm_training else None


        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.flow_optim = flow_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler
        self.flow_scheduler = flow_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip, self.args.use_dapo)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = NaiveExperienceMaker(
            actor,
            critic,
            reward_model,
            initial_model,
            self.tokenizer,
            self.data_processor,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
            reward_fn,
        )
        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, self.data_processor, buffer_limit, buffer_cpu_offload, packing_samples,
            drop_maxlen=self.args.drop_maxlen, 
            maxlen=self.args.generate_max_len + prompt_max_len,
            strategy=strategy,
        )

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
    ) -> None:
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts, labels in self.prompts_dataloader:
                experiences, accuracy_rewards_original = self.experience_maker.make_experience_list(
                    rand_prompts, labels, steps, **self.generate_kwargs
                )
                for i, experience in enumerate(experiences):
                    if i == 0:
                        output = self.tokenizer.batch_decode(
                            experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                        )
                        self.strategy.print(output)
                    self.replay_buffer.append(experience)

                if self.args.advantage_estimator != "group_norm":
                    self.replay_buffer.normalize("advantages", self.strategy)
                status = self.ppo_train(steps)
                self.replay_buffer.clear()

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps=0):
        torch.cuda.empty_cache()
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=False if self.strategy.ring_attn_group is not None else True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]
                if "update_flow_loss" in status:
                    status["update_flow_loss"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["update_flow_loss"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "adv": status["advantages"],
                        "ret": status["return"],
                        "etr": status["entropy"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }
                
                if "update_flow_loss" in status:
                    short_status["fl"] = status["update_flow_loss"]
                
                if "per_token_flow_reward" in status:
                    short_status["pfr"] = status["per_token_flow_reward"]

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    if "reward_token_id" in k:
                        if k in status_mean.keys():
                            status_mean[k] = torch.cat([status_mean[k],v])
                        else:
                            status_mean[k] = v
                    else:
                        status_mean[k] += v
            for k in status_mean.keys():
                if "reward_token_id" in k:
                    unique_ids, counts = torch.unique(status_mean[k], return_counts=True)
                    show_num = min(len(counts),50)
                    values, indices = torch.topk(counts,show_num)
                    status_mean[k] = unique_ids[indices]
                else:
                    status_mean[k] /= len(status_list)
        torch.cuda.empty_cache()
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience, global_steps)
        if self.critic is not None:
            status.update(self.training_step_critic(experience))
        return status

    def training_step_actor(self, experience: Experience, global_steps) -> Dict[str, float]:
        self.actor.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
            visual_inputs = experience.visual_inputs
            # pad seq makes the sequence a multiple of ring_attention_size.
            if self.strategy.ring_attn_group is not None:
                pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                    sequences, 
                    attention_mask, 
                    num_actions, 
                    packed_seq_lens, 
                    self.strategy.ring_attn_group
                )
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = torch.cat(experience.base_action_log_probs, dim=0).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            action_mask = experience.action_mask
            attention_mask = experience.attention_mask
            visual_inputs = experience.visual_inputs
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = experience.base_action_log_probs

        # actor loss
        action_log_probs, output, action_flow_loss, action_flow_score = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
            visual_inputs=visual_inputs
        )

        if self.args.flow_coef != 0:
            action_flow_score = normalize_flow_score(action_flow_score, action_mask, self.args)

        # compute logits entropy
        entropy = entropy_from_logits(output["logits"][:, -num_actions:])  # (bsz, response_length)
        if not self.args.packing_samples:
            batch_entropy_mean = masked_mean(entropy, action_mask, dim=-1)
            entropy_mean = ((batch_entropy_mean * experience.info["response_length"]).sum() / experience.info["response_length"].sum()).item()

        # unpad sequence ensures that pad tokens do not contribute to the loss calculation.
        if self.strategy.ring_attn_group is not None:
            assert pad_len is not None
            sequences, attention_mask, num_actions, packed_seq_lens, action_log_probs, _, _ = unpad_sequences(
                pad_len=pad_len,
                sequences=sequences,
                attention_mask=attention_mask,
                num_actions=num_actions,
                packed_seq_lens=packed_seq_lens,
                action_log_probs=action_log_probs,
                ring_attn_group=self.strategy.ring_attn_group,
            )

        # loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
        )

        if self.args.use_kl_loss:
            if self.initial_model is not None:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    experience.action_mask,
                    kl_estimator=self.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

            if not self.args.packing_samples:
                kl_mean = masked_mean(kl, experience.action_mask, dim=-1)
            else:
                # convert tensor into list of tensors so that it's easier to manipulate
                # within dataset.

                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=action_log_probs.device)

            kl_loss = kl_mean.mean()
            experience.info["kl"] = kl_loss.item()
        else:
            kl_loss = 0

        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        if global_steps > self.args.freezing_actor_steps_for_flow: 
            loss = actor_loss + aux_loss * self.args.aux_loss_coef + kl_loss * self.kl_ctl.value
            self.strategy.backward(loss, self.actor.model, self.actor_optim)
        if self.args.flow_coef != 0 and self.flow_optim:
            if self.args.experience_sampling_metrics:

                def experience_sampling(action_flow_loss, args, experience, return_mask=False):
                    experience_sampling_metrics = args.experience_sampling_metrics.split("+")
                    batch_experience_mask = torch.ones(action_flow_loss.shape[0], device=action_flow_loss.device)
                    trajectory_experience_mask = action_mask
                    if "batch_correct" in experience_sampling_metrics:
                        reward = experience.info["reward"]
                        batch_experience_mask = torch.where(reward>=1.0, 1, self.args.soft_flow)
                    if "trajectory_entropy" in experience_sampling_metrics:
                        metrics_entropy_quantile = torch.tensor(args.metrics_entropy_quantile,device=torch.cuda.current_device())
                        entropy_threshold = torch.quantile(entropy, metrics_entropy_quantile, dim=1, keepdim=True)
                        trajectory_experience_mask = torch.where(entropy >= entropy_threshold, 1, self.args.soft_flow)
                        trajectory_experience_mask = trajectory_experience_mask * action_mask

                    action_flow_loss = masked_mean(action_flow_loss, trajectory_experience_mask, dim=-1)
                    action_flow_loss = masked_mean(action_flow_loss, batch_experience_mask, dim=0)
                    
                    if return_mask:
                        return action_flow_loss, batch_experience_mask, trajectory_experience_mask
                    return action_flow_loss
                
                action_flow_loss, batch_experience_mask, trajectory_experience_mask = experience_sampling(action_flow_loss, self.args, experience, return_mask=True)

            else:
                action_flow_loss = masked_mean(action_flow_loss, action_mask, dim=-1)
                action_flow_loss = action_flow_loss.mean()
            

            experience.info["update_flow_loss"] = action_flow_loss.item()
            action_flow_loss.requires_grad = True
            self.strategy.backward(action_flow_loss, self.actor.flow, self.flow_optim)

        # ptx loss
        if global_steps > self.args.freezing_actor_steps_for_flow: 
            if self.pretrain_dataloader is not None:
                data = next(self.pretrain_dataloader)
                inputs = data[1].squeeze(1).to(torch.cuda.current_device())
                attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
                label = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.ptx_loss_fn.IGNORE_INDEX,
                )

                output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
                ptx_log_probs = output["logits"]

                # loss function
                ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0
                loss = ptx_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(self.ptx_coef * loss, self.actor.model, self.actor_optim)

        if global_steps > self.args.freezing_actor_steps_for_flow: 
            self.strategy.optimizer_step(self.actor_optim, self.actor.model, self.actor_scheduler, name="actor")
            if self.ema_model:
                self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
        if self.flow_optim:
            self.strategy.optimizer_step(self.flow_optim, self.actor.flow, self.flow_scheduler, name="flow")

        # status
        status = {
            "policy_loss": actor_loss.item(), 
            "actor_lr": self.actor_scheduler.get_last_lr()[0], 
            "entropy": entropy_mean, 
            "advantages": masked_mean(advantages, action_mask, dim=-1).mean().item()
        }


        if self.args.flow_coef != 0:
            action_flow_reward, reward_indices = postprocess_flow_score(action_flow_score, self.args, return_indices=True)
                
            if reward_indices:
                if len(reward_indices) == 2:
                    status["reward_token_id_pos"] = torch.gather(sequences[:, -num_actions:]*action_mask, 1, reward_indices[0])
                    status["reward_token_id_pos"] = torch.gather(sequences[:, -num_actions:]*action_mask, 1, reward_indices[1])
                else:
                    status["reward_token_id"] = torch.gather(sequences[:, -num_actions:]*action_mask, 1, reward_indices[0])
            
            status["num_flow_reward_tokens"] = torch.count_nonzero(action_flow_reward, dim=1).float().mean().item()
            status["per_token_flow_reward"] = self.args.flow_coef * (action_flow_reward.abs().sum(axis=1) / torch.count_nonzero(action_flow_reward, dim=1)).mean().item()
            status["total_flow_reward"] = self.args.flow_coef * (action_flow_reward.abs().sum(axis=1)).mean().item()
            # correct_experience_mask = reward>=1.0
            # incorrect_experience_mask = reward<1.0
            # status["total_matching_reward_correct"] = self.args.matching_coef * (matching_clip[correct_experience_mask].abs().sum(axis=1)).mean().item()
            # status["total_matching_reward_incorrect"] = self.args.matching_coef * (matching_clip[incorrect_experience_mask].abs().sum(axis=1)).mean().item()
        
        
            if self.flow_optim and self.args.experience_sampling_metrics:
                if "batch" in self.args.experience_sampling_metrics:
                    if "correct" in self.args.experience_sampling_metrics:
                        status["update_batch_ratio"] = batch_experience_mask.sum().item() / batch_experience_mask.shape[0]
                if "trajectory" in self.args.experience_sampling_metrics:
                    status["update_trajectory_ratio"] = (trajectory_experience_mask.sum(1) / action_mask.sum(1)).mean().item()

        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == "kl" or k=="update_flow_loss":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status

    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
            visual_inputs = experience.visual_inputs
            # pad seq makes the sequence len a multiple of ring_attention_size.
            if self.strategy.ring_attn_group is not None:
                pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                    sequences, 
                    attention_mask, 
                    num_actions, 
                    packed_seq_lens, 
                    self.strategy.ring_attn_group
                )

        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            visual_inputs = experience.visual_inputs

        # critic loss
        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            values_allgather=True,
            packed_seq_lens=packed_seq_lens,
            visual_inputs=visual_inputs,
        )
        # unpad sequence ensures that pad tokens do not contribute to the loss calculation
        if self.strategy.ring_attn_group is not None:
            assert pad_len is not None
            sequences, attention_mask, num_actions, packed_seq_lens, _, values, _ = unpad_sequences(
                pad_len=pad_len,
                sequences=sequences,
                attention_mask=attention_mask,
                num_actions=num_actions,
                packed_seq_lens=packed_seq_lens,
                values=values,
                ring_attn_group=self.strategy.ring_attn_group,
            )

        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            response_length_list = torch.tensor(self.experience_maker.response_length_list)
            gathered_response_length_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_response_length_list, response_length_list)
            response_length_list = torch.cat(gathered_response_length_list).tolist()
            assert len(response_length_list) > 0
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                from wandb import Histogram
                response_length_list = Histogram(response_length_list)
                logs["response_length_dist"] = response_length_list
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    if "reward_token_id" in k:
                        self._tensorboard.add_text(f"train/{k}", self.tokenizer.decode(v), global_step)
                    else:
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)
            pass
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )
            if self.critic is not None:
                self.strategy.save_ckpt(
                    self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
                )

        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            if args.mllm_training:
                self.strategy.save_model(self.actor, self.processor or self.tokenizer, save_path)
                if args.save_flow:
                    self.strategy.save_model(self.actor.flow, self.processor or self.tokenizer, save_path+"_flow")
            elif args.llm_training:
                self.strategy.save_model(self.actor, self.tokenizer, save_path)
