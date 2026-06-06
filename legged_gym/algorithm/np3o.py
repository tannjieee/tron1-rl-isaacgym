import math

import torch
import torch.nn as nn

from .ppo import PPO
from .constraint_rollout_storage import ConstraintRolloutStorage


class NP3O(PPO):
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
        cost_dof_pos_limit=1.0,
        cost_dof_vel_limit=0.2,
        cost_torque_limit=0.2,
        cost_collision=1.0,
        cost_termination_contact=2.0,
        cost_fall=2.0,
        cost_power_limit=0.2,
        cost_wheel_vel_limit=0.1,
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
        self.soft_dof_vel_limit = soft_dof_vel_limit
        self.soft_torque_limit = soft_torque_limit
        self.contact_force_threshold = float(contact_force_threshold)
        self.termination_contact_force_threshold = float(termination_contact_force_threshold)
        self.fall_projected_gravity_z = float(fall_projected_gravity_z)
        self.wheel_name_keys = tuple(wheel_name_keys)
        self.mean_batch_cost = torch.tensor(0.0, device=self.device)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, obs_history_shape, commands_shape, action_shape):
        self.storage = ConstraintRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    def _col(self, x, n):
        if x is None:
            return torch.zeros(n, 1, device=self.device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float, device=self.device)
        else:
            x = x.to(self.device, dtype=torch.float)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x.view(n, -1)[:, :1]

    def _safe(self, x):
        return torch.clamp(torch.abs(x), min=1.0e-6)

    def _soft(self, env, name, default):
        v = getattr(self, name, None)
        if v is not None:
            return float(v)
        rewards_cfg = getattr(getattr(env, "cfg", None), "rewards", None)
        return float(getattr(rewards_cfg, name, default)) if rewards_cfg is not None else default

    def _wheel_mask(self, env):
        names = getattr(env, "dof_names", [])
        mask = [any(k.lower() in name.lower() for k in self.wheel_name_keys) for name in names]
        if len(mask) == 0 or not any(mask):
            return None
        return torch.tensor(mask, dtype=torch.bool, device=self.device)

    def _compute_cost_from_env(self, infos, env):
        if infos is not None and "cost" in infos:
            return self._col(infos["cost"], self.num_group)
        if env is not None and hasattr(env, "cost_buf"):
            return self._col(env.cost_buf, self.num_group)
        if env is None:
            return torch.zeros(self.num_group, 1, device=self.device)

        n = getattr(env, "num_envs", self.num_group)
        cost = torch.zeros(n, device=self.device)

        dof_pos = getattr(env, "dof_pos", None)
        dof_vel = getattr(env, "dof_vel", None)
        torques = getattr(env, "torques", None)
        pos_lim = getattr(env, "dof_pos_limits", None)
        vel_lim = getattr(env, "dof_vel_limits", None)
        tau_lim = getattr(env, "torque_limits", None)

        if dof_pos is not None and pos_lim is not None and self.cost_dof_pos_limit > 0.0:
            dof_pos = dof_pos.to(self.device)
            pos_lim = pos_lim.to(self.device)
            half = 0.5 * (pos_lim[:, 1] - pos_lim[:, 0])
            excess = torch.clamp(pos_lim[:, 0] - dof_pos, min=0.0) + torch.clamp(dof_pos - pos_lim[:, 1], min=0.0)
            cost += self.cost_dof_pos_limit * torch.sum(excess / self._safe(half), dim=1)

        if dof_vel is not None and vel_lim is not None and self.cost_dof_vel_limit > 0.0:
            dof_vel = dof_vel.to(self.device)
            vel_lim = vel_lim.to(self.device)
            lim = self._soft(env, "soft_dof_vel_limit", 1.0) * vel_lim
            excess = torch.clamp(torch.abs(dof_vel) - lim, min=0.0)
            cost += self.cost_dof_vel_limit * torch.sum(excess / self._safe(vel_lim), dim=1)

        if torques is not None and tau_lim is not None and self.cost_torque_limit > 0.0:
            torques = torques.to(self.device)
            tau_lim = tau_lim.to(self.device)
            lim = self._soft(env, "soft_torque_limit", 0.8) * tau_lim
            excess = torch.clamp(torch.abs(torques) - lim, min=0.0)
            cost += self.cost_torque_limit * torch.sum(excess / self._safe(tau_lim), dim=1)

        if dof_vel is not None and vel_lim is not None and self.cost_wheel_vel_limit > 0.0:
            mask = self._wheel_mask(env)
            if mask is not None:
                excess = torch.clamp(torch.abs(dof_vel[:, mask]) - vel_lim[mask], min=0.0)
                cost += self.cost_wheel_vel_limit * torch.sum(excess / self._safe(vel_lim[mask]), dim=1)

        cf = getattr(env, "contact_forces", None)
        idx = getattr(env, "penalised_contact_indices", None)
        if cf is not None and idx is not None and len(idx) > 0 and self.cost_collision > 0.0:
            cost += self.cost_collision * torch.sum(
                torch.norm(cf[:, idx, :], dim=-1) > self.contact_force_threshold, dim=1
            ).float()

        tidx = getattr(env, "termination_contact_indices", None)
        if cf is not None and tidx is not None and len(tidx) > 0 and self.cost_termination_contact > 0.0:
            cost += self.cost_termination_contact * torch.any(
                torch.norm(cf[:, tidx, :], dim=-1) > self.termination_contact_force_threshold, dim=1
            ).float()

        g = getattr(env, "projected_gravity", None)
        if g is not None and self.cost_fall > 0.0:
            cost += self.cost_fall * (g[:, 2].to(self.device) > self.fall_projected_gravity_z).float()

        p = getattr(env, "power_limit_out_buf", None)
        if p is not None and self.cost_power_limit > 0.0:
            cost += self.cost_power_limit * p.to(self.device).float()

        if infos is not None:
            infos["cost"] = cost.detach()
        return cost.view(n, 1)

    def act(self, obs, obs_history, commands, critic_obs):
        actions = super().act(obs, obs_history, commands, critic_obs)
        self.transition.cost_values = self.actor_critic.evaluate_cost(self.transition.critic_obs).detach()
        return actions

    def process_env_step(self, rewards, dones, infos, next_obs=None, env=None):
        self.transition.rewards = rewards.clone()
        self.transition.costs = self._compute_cost_from_env(infos, env).clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            time_out = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_out, 1)
            self.transition.costs += self.gamma * self.transition.cost_values * time_out
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        last_cost_values = self.actor_critic.evaluate_cost(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        self.storage.compute_cost_returns(last_cost_values, self.gamma, self.lam, self.normalize_cost_advantage)
        self.mean_batch_cost = self.storage.costs.mean().detach()
        if self.use_adaptive_penalty:
            violation = (self.mean_batch_cost - self.cost_limit).item()
            self.penalty_coef *= math.exp(self.penalty_lr * violation)
            self.penalty_coef = min(max(self.penalty_coef, self.min_penalty_coef), self.max_penalty_coef)

    def update(self):
        n = 0
        mean_value_loss = mean_cost_value_loss = mean_surrogate_loss = 0.0
        mean_cost_surrogate_loss = mean_penalty_loss = mean_kl = 0.0
        generator = self.storage.constraint_mini_batch_generator(self.num_group, self.num_mini_batches, self.num_learning_epochs)
        for (obs_b, critic_obs_b, hist_b, _, cmd_b, act_b, val_b, adv_b, ret_b,
             old_logp_b, old_mu_b, old_sigma_b, cost_val_b, cost_adv_b, cost_ret_b) in generator:
            enc_b = self.encoder.encode(hist_b)
            self.actor_critic.act(torch.cat((enc_b, obs_b, cmd_b), dim=-1))
            logp_b = self.actor_critic.get_actions_log_prob(act_b)
            value_b = self.actor_critic.evaluate(critic_obs_b)
            cost_value_b = self.actor_critic.evaluate_cost(critic_obs_b)
            mu_b = self.actor_critic.action_mean
            sigma_b = self.actor_critic.action_std
            entropy_b = self.actor_critic.entropy

            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_b / old_sigma_b + 1.0e-5)
                    + (torch.square(old_sigma_b) + torch.square(old_mu_b - mu_b)) / (2.0 * torch.square(sigma_b))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            if self.desired_kl is not None and self.early_stop and kl_mean > self.desired_kl * 1.5:
                break

            ratio = torch.exp(logp_b - torch.squeeze(old_logp_b))
            surr = -torch.squeeze(adv_b) * ratio
            surr_clip = -torch.squeeze(adv_b) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surr, surr_clip).mean()

            cost_surr = torch.squeeze(cost_adv_b) * ratio
            cost_surr_clip = torch.squeeze(cost_adv_b) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            cost_surrogate_loss = torch.max(cost_surr, cost_surr_clip).mean()
            penalty_loss = self.penalty_coef * torch.relu(cost_surrogate_loss + self.mean_batch_cost - self.cost_limit)

            if self.use_clipped_value_loss:
                value_clip = val_b + (value_b - val_b).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((value_b - ret_b).pow(2), (value_clip - ret_b).pow(2)).mean()
                cost_value_clip = cost_val_b + (cost_value_b - cost_val_b).clamp(-self.clip_param, self.clip_param)
                cost_value_loss = torch.max((cost_value_b - cost_ret_b).pow(2), (cost_value_clip - cost_ret_b).pow(2)).mean()
            else:
                value_loss = (ret_b - value_b).pow(2).mean()
                cost_value_loss = (cost_ret_b - cost_value_b).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss + self.cost_value_loss_coef * cost_value_loss + penalty_loss - self.entropy_coef * entropy_b.mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            n += 1
            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_cost_surrogate_loss += cost_surrogate_loss.item()
            mean_penalty_loss += penalty_loss.item()
            mean_kl += kl_mean.item()

        mean_extra_loss = self._update_encoder()
        denom = max(n, 1)
        self.storage.clear()
        return (
            mean_value_loss / denom,
            mean_extra_loss,
            mean_surrogate_loss / denom,
            mean_kl / denom,
            mean_cost_value_loss / denom,
            mean_cost_surrogate_loss / denom,
            mean_penalty_loss / denom,
            self.mean_batch_cost.item(),
            self.penalty_coef,
        )

    def _update_encoder(self):
        n = 0
        mean_loss = 0.0
        if self.extra_optimizer is None:
            return 0.0
        generator = self.storage.encoder_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for _, critic_obs_b, hist_b in generator:
            if self.encoder.is_mlp_encoder:
                self.encoder.encode(hist_b)
                enc_b = self.encoder.get_encoder_out()
                loss = (enc_b[:, 0:3] - critic_obs_b[:, 0:3]).pow(2).mean()
            else:
                loss = torch.zeros(1, device=self.device).mean()
            self.extra_optimizer.zero_grad()
            loss.backward()
            self.extra_optimizer.step()
            n += 1
            mean_loss += loss.item()
        return mean_loss / max(n, 1)
