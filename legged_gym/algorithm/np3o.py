# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# NP3O/P3O-style constrained PPO extension for wheeled-foot locomotion.
# The implementation is intentionally kept close to the local PPO class so that
# existing legged_gym tasks can switch between PPO and NP3O from config only.

import math

import torch
import torch.nn as nn

from .ppo import PPO


class NP3O(PPO):
    """Normalized/Penalized PPO for cost-constrained locomotion.

    Costs are collected from ``infos['cost']``, ``env.cost_buf`` or computed from
    common legged_gym tensors such as joint limits, torque limits, contacts and
    projected gravity. The reward update remains PPO; the cost update adds a
    clipped cost surrogate and a cost critic.
    """

    def __init__(
        self,
        *args,
        cost_limit=0.25,
        penalty_coef=1.0,
        min_penalty_coef=0.0,
        max_penalty_coef=100.0,
        penalty_lr=0.05,
        use_adaptive_penalty=True,
        cost_value_loss_coef=1.0,
        normalize_cost_advantage=False,
        # Cost term switches / scales. These defaults are conservative and are
        # intended to prevent clearly unsafe hardware-limit violations while not
        # killing early stair-climbing exploration.
        cost_dof_pos_limit=1.0,
        cost_dof_vel_limit=0.2,
        cost_torque_limit=0.2,
        cost_collision=1.0,
        cost_termination_contact=2.0,
        cost_fall=2.0,
        cost_power_limit=0.2,
        cost_wheel_vel_limit=0.1,
        soft_dof_pos_limit=None,
        soft_dof_vel_limit=None,
        soft_torque_limit=None,
        contact_force_threshold=1.0,
        termination_contact_force_threshold=10.0,
        fall_projected_gravity_z=-0.1,
        wheel_name_keys=("wheel",),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cost_limit = float(cost_limit)
        self.penalty_coef = float(penalty_coef)
        self.min_penalty_coef = float(min_penalty_coef)
        self.max_penalty_coef = float(max_penalty_coef)
        self.penalty_lr = float(penalty_lr)
        self.use_adaptive_penalty = bool(use_adaptive_penalty)
        self.cost_value_loss_coef = float(cost_value_loss_coef)
        self.normalize_cost_advantage = bool(normalize_cost_advantage)

        self.cost_dof_pos_limit = float(cost_dof_pos_limit)
        self.cost_dof_vel_limit = float(cost_dof_vel_limit)
        self.cost_torque_limit = float(cost_torque_limit)
        self.cost_collision = float(cost_collision)
        self.cost_termination_contact = float(cost_termination_contact)
        self.cost_fall = float(cost_fall)
        self.cost_power_limit = float(cost_power_limit)
        self.cost_wheel_vel_limit = float(cost_wheel_vel_limit)
        self.soft_dof_pos_limit = soft_dof_pos_limit
        self.soft_dof_vel_limit = soft_dof_vel_limit
        self.soft_torque_limit = soft_torque_limit
        self.contact_force_threshold = float(contact_force_threshold)
        self.termination_contact_force_threshold = float(termination_contact_force_threshold)
        self.fall_projected_gravity_z = float(fall_projected_gravity_z)
        self.wheel_name_keys = tuple(wheel_name_keys)

        self.mean_batch_cost = torch.tensor(0.0, device=self.device)

    def _as_column(self, tensor, num_envs):
        if tensor is None:
            return torch.zeros(num_envs, 1, device=self.device)
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float, device=self.device)
        else:
            tensor = tensor.to(device=self.device, dtype=torch.float)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        return tensor.view(num_envs, -1)[:, :1]

    def _safe_denominator(self, x):
        return torch.clamp(torch.abs(x), min=1.0e-6)

    def _get_cfg_soft_limit(self, env, name, default):
        if getattr(self, name, None) is not None:
            return float(getattr(self, name))
        rewards_cfg = getattr(getattr(env, "cfg", None), "rewards", None)
        if rewards_cfg is None:
            return default
        return float(getattr(rewards_cfg, name, default))

    def _wheel_dof_mask(self, env):
        names = getattr(env, "dof_names", [])
        if not names:
            return None
        mask = [any(key.lower() in name.lower() for key in self.wheel_name_keys) for name in names]
        if not any(mask):
            return None
        return torch.tensor(mask, dtype=torch.bool, device=self.device)

    def _compute_cost_from_env(self, infos, env):
        if infos is not None and "cost" in infos:
            return self._as_column(infos["cost"], self.num_group)
        if env is not None and hasattr(env, "cost_buf"):
            return self._as_column(env.cost_buf, self.num_group)
        if env is None:
            return torch.zeros(self.num_group, 1, device=self.device)

        num_envs = getattr(env, "num_envs", self.num_group)
        cost = torch.zeros(num_envs, device=self.device)
        cost_terms = {}

        dof_pos = getattr(env, "dof_pos", None)
        dof_vel = getattr(env, "dof_vel", None)
        torques = getattr(env, "torques", None)
        dof_pos_limits = getattr(env, "dof_pos_limits", None)
        dof_vel_limits = getattr(env, "dof_vel_limits", None)
        torque_limits = getattr(env, "torque_limits", None)

        if dof_pos is not None and dof_pos_limits is not None and self.cost_dof_pos_limit > 0.0:
            dof_pos = dof_pos.to(self.device)
            dof_pos_limits = dof_pos_limits.to(self.device)
            soft = self._get_cfg_soft_limit(env, "soft_dof_pos_limit", 0.95)
            mid = 0.5 * (dof_pos_limits[:, 0] + dof_pos_limits[:, 1])
            half_range = 0.5 * (dof_pos_limits[:, 1] - dof_pos_limits[:, 0])
            lower = mid - soft * half_range
            upper = mid + soft * half_range
            excess = torch.clamp(lower - dof_pos, min=0.0) + torch.clamp(dof_pos - upper, min=0.0)
            term = torch.sum(excess / self._safe_denominator(half_range), dim=1)
            cost_terms["dof_pos_limit"] = term
            cost += self.cost_dof_pos_limit * term

        if dof_vel is not None and dof_vel_limits is not None and self.cost_dof_vel_limit > 0.0:
            dof_vel = dof_vel.to(self.device)
            dof_vel_limits = dof_vel_limits.to(self.device)
            soft = self._get_cfg_soft_limit(env, "soft_dof_vel_limit", 1.0)
            limit = soft * dof_vel_limits
            excess = torch.clamp(torch.abs(dof_vel) - limit, min=0.0)
            term = torch.sum(excess / self._safe_denominator(dof_vel_limits), dim=1)
            cost_terms["dof_vel_limit"] = term
            cost += self.cost_dof_vel_limit * term

        if torques is not None and torque_limits is not None and self.cost_torque_limit > 0.0:
            torques = torques.to(self.device)
            torque_limits = torque_limits.to(self.device)
            soft = self._get_cfg_soft_limit(env, "soft_torque_limit", 0.8)
            limit = soft * torque_limits
            excess = torch.clamp(torch.abs(torques) - limit, min=0.0)
            term = torch.sum(excess / self._safe_denominator(torque_limits), dim=1)
            cost_terms["torque_limit"] = term
            cost += self.cost_torque_limit * term

        if dof_vel is not None and dof_vel_limits is not None and self.cost_wheel_vel_limit > 0.0:
            wheel_mask = self._wheel_dof_mask(env)
            if wheel_mask is not None:
                wheel_vel = dof_vel[:, wheel_mask].to(self.device)
                wheel_limits = dof_vel_limits[wheel_mask].to(self.device)
                excess = torch.clamp(torch.abs(wheel_vel) - wheel_limits, min=0.0)
                term = torch.sum(excess / self._safe_denominator(wheel_limits), dim=1)
                cost_terms["wheel_vel_limit"] = term
                cost += self.cost_wheel_vel_limit * term

        contact_forces = getattr(env, "contact_forces", None)
        penalised_contact_indices = getattr(env, "penalised_contact_indices", None)
        if (
            contact_forces is not None
            and penalised_contact_indices is not None
            and len(penalised_contact_indices) > 0
            and self.cost_collision > 0.0
        ):
            term = torch.sum(
                torch.norm(contact_forces[:, penalised_contact_indices, :], dim=-1)
                > self.contact_force_threshold,
                dim=1,
            ).float()
            cost_terms["collision"] = term
            cost += self.cost_collision * term

        termination_contact_indices = getattr(env, "termination_contact_indices", None)
        if (
            contact_forces is not None
            and termination_contact_indices is not None
            and len(termination_contact_indices) > 0
            and self.cost_termination_contact > 0.0
        ):
            term = torch.any(
                torch.norm(contact_forces[:, termination_contact_indices, :], dim=-1)
                > self.termination_contact_force_threshold,
                dim=1,
            ).float()
            cost_terms["termination_contact"] = term
            cost += self.cost_termination_contact * term

        projected_gravity = getattr(env, "projected_gravity", None)
        if projected_gravity is not None and self.cost_fall > 0.0:
            term = (projected_gravity[:, 2].to(self.device) > self.fall_projected_gravity_z).float()
            cost_terms["fall"] = term
            cost += self.cost_fall * term

        power_limit_out_buf = getattr(env, "power_limit_out_buf", None)
        if power_limit_out_buf is not None and self.cost_power_limit > 0.0:
            term = power_limit_out_buf.to(self.device).float()
            cost_terms["power_limit"] = term
            cost += self.cost_power_limit * term

        if infos is not None:
            infos["cost"] = cost.detach()
            infos["cost_terms"] = {k: v.detach() for k, v in cost_terms.items()}
        return cost.view(num_envs, 1)

    def act(self, obs, obs_history, commands, critic_obs):
        actions = super().act(obs, obs_history, commands, critic_obs)
        self.transition.cost_values = self.actor_critic.evaluate_cost(
            self.transition.critic_obs
        ).detach()
        return actions

    def process_env_step(self, rewards, dones, infos, next_obs=None, env=None):
        costs = self._compute_cost_from_env(infos, env)
        self.transition.costs = costs.clone()
        if "time_outs" in infos:
            self.transition.costs += self.gamma * torch.squeeze(
                self.transition.cost_values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            ).view(-1, 1)
        super().process_env_step(rewards, dones, infos, next_obs)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        last_cost_values = self.actor_critic.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        self.storage.compute_cost_returns(
            last_cost_values,
            self.gamma,
            self.lam,
            normalize=self.normalize_cost_advantage,
        )
        self.mean_batch_cost = self.storage.costs.mean().detach()

        if self.use_adaptive_penalty:
            violation = (self.mean_batch_cost - self.cost_limit).item()
            self.penalty_coef *= math.exp(self.penalty_lr * violation)
            self.penalty_coef = float(
                min(max(self.penalty_coef, self.min_penalty_coef), self.max_penalty_coef)
            )

    def update(self):
        num_updates = 0
        mean_value_loss = 0.0
        mean_cost_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_cost_surrogate_loss = 0.0
        mean_penalty_loss = 0.0
        mean_kl = 0.0
        generator = self.storage.constraint_mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch,
            _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            target_cost_values_batch,
            cost_advantages_batch,
            cost_returns_batch,
        ) in generator:
            encoder_out_batch = self.encoder.encode(obs_history_batch)
            commands_batch = group_commands_batch
            self.actor_critic.act(
                torch.cat((encoder_out_batch, obs_batch, commands_batch), dim=-1)
            )

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            cost_value_batch = self.actor_critic.evaluate_cost(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.desired_kl != None and self.early_stop:
                if kl_mean > self.desired_kl * 1.5:
                    print("early stop, num_updates =", num_updates)
                    break

            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )

            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            cost_surrogate = torch.squeeze(cost_advantages_batch) * ratio
            cost_surrogate_clipped = torch.squeeze(cost_advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            cost_surrogate_loss = torch.max(
                cost_surrogate, cost_surrogate_clipped
            ).mean()
            cost_violation = self.mean_batch_cost - self.cost_limit
            penalty_loss = self.penalty_coef * torch.relu(
                cost_surrogate_loss + cost_violation
            )

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()

                cost_value_clipped = target_cost_values_batch + (
                    cost_value_batch - target_cost_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                cost_value_losses = (cost_value_batch - cost_returns_batch).pow(2)
                cost_value_losses_clipped = (
                    cost_value_clipped - cost_returns_batch
                ).pow(2)
                cost_value_loss = torch.max(
                    cost_value_losses, cost_value_losses_clipped
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

            entropy_batch_mean = entropy_batch.mean()
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * cost_value_loss
                + penalty_loss
                - self.entropy_coef * entropy_batch_mean
            )

            if self.anneal_lr:
                frac = 1.0 - num_updates / (
                    self.num_learning_epochs * self.num_mini_batches
                )
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_cost_surrogate_loss += cost_surrogate_loss.item()
            mean_penalty_loss += penalty_loss.item()
            mean_kl += kl_mean.item()

        num_updates_extra = 0
        mean_extra_loss = 0.0
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for next_obs_batch, critic_obs_batch, obs_history_batch in generator:
                if self.encoder.is_mlp_encoder:
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()
                    extra_loss = (
                        (encode_batch[:, 0:3] - critic_obs_batch[:, 0:3]).pow(2).mean()
                    )
                else:
                    extra_loss = torch.zeros(1, device=self.device).mean()

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                self.extra_optimizer.step()

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()

        mean_value_loss /= max(num_updates, 1)
        mean_cost_value_loss /= max(num_updates, 1)
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
        mean_surrogate_loss /= max(num_updates, 1)
        mean_cost_surrogate_loss /= max(num_updates, 1)
        mean_penalty_loss /= max(num_updates, 1)
        mean_kl /= max(num_updates, 1)
        mean_cost = self.mean_batch_cost.item()
        self.storage.clear()

        return (
            mean_value_loss,
            mean_extra_loss,
            mean_surrogate_loss,
            mean_kl,
            mean_cost_value_loss,
            mean_cost_surrogate_loss,
            mean_penalty_loss,
            mean_cost,
            self.penalty_coef,
        )
