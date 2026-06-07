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
        cost_k_values=None,
        cost_d_values=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_costs = max(int(getattr(self.actor_critic, "num_costs", 1)), 1)
        self.cost_limit = float(cost_limit)
        self.penalty_coef = float(penalty_coef)
        self.min_penalty_coef = float(min_penalty_coef)
        self.max_penalty_coef = float(max_penalty_coef)
        self.penalty_lr = float(penalty_lr)
        self.use_adaptive_penalty = bool(use_adaptive_penalty)
        self.cost_value_loss_coef = float(cost_value_loss_coef)
        self.normalize_cost_advantage = bool(normalize_cost_advantage)
        self.cost_k_values = self._cost_vector(cost_k_values, default_value=1.0)
        self.cost_d_values = self._cost_vector(
            cost_d_values,
            default_value=self.cost_limit / self.num_costs,
        )
        self.mean_step_cost = torch.tensor(0.0, device=self.device)
        self.mean_discounted_cost = torch.tensor(0.0, device=self.device)
        self.mean_step_cost_by_type = torch.zeros(self.num_costs, device=self.device)
        self.mean_discounted_cost_by_type = torch.zeros(self.num_costs, device=self.device)
        self.mean_batch_cost = torch.tensor(0.0, device=self.device)

    def _cost_vector(self, values, default_value):
        if values is None:
            vector = torch.full(
                (self.num_costs,),
                default_value,
                dtype=torch.float,
                device=self.device,
            )
        elif not torch.is_tensor(values):
            vector = torch.as_tensor(values, dtype=torch.float, device=self.device).view(-1)
        else:
            vector = values.to(self.device, dtype=torch.float).view(-1)

        if vector.numel() == 0:
            vector = torch.full(
                (self.num_costs,),
                default_value,
                dtype=torch.float,
                device=self.device,
            )
        elif vector.numel() == 1 and self.num_costs > 1:
            vector = vector.repeat(self.num_costs)
        elif vector.numel() < self.num_costs:
            padding = torch.full(
                (self.num_costs - vector.numel(),),
                default_value,
                dtype=torch.float,
                device=self.device,
            )
            vector = torch.cat((vector, padding), dim=0)
        elif vector.numel() > self.num_costs:
            vector = vector[:self.num_costs]
        return vector.view(1, self.num_costs)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, obs_history_shape, commands_shape, action_shape):
        self.storage = ConstraintRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
            obs_history_shape, commands_shape, action_shape, self.device,
            cost_shape=[self.num_costs],
        )

    def _col(self, x, n):
        if x is None:
            return torch.zeros(n, self.num_costs, device=self.device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float, device=self.device)
        else:
            x = x.to(self.device, dtype=torch.float)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = x.view(n, -1)
        if x.shape[1] == self.num_costs:
            return x
        if x.shape[1] == 1:
            if self.num_costs == 1:
                return x
            return x.repeat(1, self.num_costs)
        if x.shape[1] < self.num_costs:
            padding = torch.zeros(n, self.num_costs - x.shape[1], device=self.device)
            return torch.cat((x, padding), dim=1)
        return x[:, :self.num_costs]

    def _aggregate_cost(self, cost):
        return torch.sum(cost * self.cost_k_values.squeeze(0), dim=-1)

    def _cost_violation(self, cost_surrogate_loss):
        return (
            (1.0 - self.gamma) * cost_surrogate_loss
            + self.mean_discounted_cost_by_type
            - self.cost_d_values.squeeze(0)
        )

    def _compute_cost_from_env(self, infos, env):
        if infos is not None and "cost_terms" in infos:
            return self._col(infos["cost_terms"], self.num_group)
        if env is not None and hasattr(env, "cost_terms_buf") and env.cost_terms_buf.shape[1] > 0:
            return self._col(env.cost_terms_buf, getattr(env, "num_envs", self.num_group))
        if infos is not None and "cost" in infos:
            return self._col(infos["cost"], self.num_group)
        if env is None:
            return torch.zeros(self.num_group, self.num_costs, device=self.device)

        n = getattr(env, "num_envs", self.num_group)
        if hasattr(env, "cost_buf"):
            cost = self._col(env.cost_buf, n)
            if infos is not None:
                infos["cost"] = cost.detach()
                if hasattr(env, "cost_terms_buf"):
                    infos["cost_terms"] = env.cost_terms_buf.detach()
            return cost

        return torch.zeros(n, self.num_costs, device=self.device)

    def act(self, obs, obs_history, commands, critic_obs, vicreg_view1=None, vicreg_view2=None):
        actions = super().act(
            obs,
            obs_history,
            commands,
            critic_obs,
            vicreg_view1=vicreg_view1,
            vicreg_view2=vicreg_view2,
        )
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
        self.mean_step_cost_by_type = self.storage.costs.mean(dim=(0, 1)).detach()
        self.mean_discounted_cost_by_type = (
            (1.0 - self.gamma) * self.storage.cost_returns[0].mean(dim=0)
        ).detach()
        self.mean_step_cost = self._aggregate_cost(self.mean_step_cost_by_type).detach()
        self.mean_discounted_cost = self._aggregate_cost(self.mean_discounted_cost_by_type).detach()
        self.mean_batch_cost = self.mean_discounted_cost
        if self.use_adaptive_penalty:
            violation = torch.sum(
                self.cost_k_values.squeeze(0)
                * (self.mean_discounted_cost_by_type - self.cost_d_values.squeeze(0))
            ).item()
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

            cost_surr = cost_adv_b * ratio.unsqueeze(-1)
            cost_surr_clip = cost_adv_b * torch.clamp(
                ratio.unsqueeze(-1),
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            cost_surrogate_loss = torch.max(cost_surr, cost_surr_clip).mean(dim=0)
            cost_violation = self._cost_violation(cost_surrogate_loss)
            penalty_loss = self.penalty_coef * torch.sum(
                self.cost_k_values.squeeze(0) * torch.relu(cost_violation)
            )

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
            mean_cost_surrogate_loss += torch.mean(cost_surrogate_loss).item()
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
            self.mean_discounted_cost.item(),
            self.penalty_coef,
        )

    def _update_encoder(self):
        n = 0
        mean_loss = 0.0
        if self.extra_optimizer is None:
            return 0.0
        generator = self.storage.encoder_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for _, critic_obs_b, hist_b, view1_b, view2_b in generator:
            loss = self._encoder_auxiliary_loss(critic_obs_b, hist_b, view1_b, view2_b)
            self.extra_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            self.extra_optimizer.step()
            n += 1
            mean_loss += loss.item()
        return mean_loss / max(n, 1)
