import torch

from .rollout_storage import RolloutStorage


class ConstraintRolloutStorage(RolloutStorage):
    def __init__(self, *args, cost_shape=None, **kwargs):
        super().__init__(*args, **kwargs)
        if cost_shape is None:
            cost_shape = [1]
        self.cost_shape = cost_shape
        self.costs = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *cost_shape, device=self.device
        )
        self.cost_values = torch.zeros_like(self.costs)
        self.cost_returns = torch.zeros_like(self.costs)
        self.cost_advantages = torch.zeros_like(self.costs)

    def add_transitions(self, transition):
        step = self.step
        if getattr(transition, "costs", None) is not None:
            self.costs[step].copy_(transition.costs.view(self.num_envs, *self.cost_shape))
        if getattr(transition, "cost_values", None) is not None:
            self.cost_values[step].copy_(transition.cost_values)
        super().add_transitions(transition)

    def compute_cost_returns(self, last_cost_values, gamma, lam, normalize=False):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_cost_values
            else:
                next_values = self.cost_values[step + 1]
            not_done = 1.0 - self.dones[step].float()
            delta = self.costs[step] + not_done * gamma * next_values - self.cost_values[step]
            advantage = delta + not_done * gamma * lam * advantage
            self.cost_returns[step] = advantage + self.cost_values[step]
        self.cost_advantages = self.cost_returns - self.cost_values
        if normalize:
            mean = self.cost_advantages.mean(dim=(0, 1), keepdim=True)
            std = self.cost_advantages.std(dim=(0, 1), keepdim=True)
            self.cost_advantages = (self.cost_advantages - mean) / (std + 1e-8)

    def _flat(self, tensor, group_idx):
        return tensor[:, group_idx, :].flatten(0, 1)

    def constraint_mini_batch_generator(self, num_group, num_mini_batches, num_epochs=8):
        batch_size = num_group * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        group_idx = torch.arange(0, num_group, device=self.device)
        arrays = (
            self._flat(self.observations, group_idx),
            self._flat(self.critic_obs, group_idx),
            self._flat(self.observation_history, group_idx),
            self._flat(self.observation_history, group_idx),
            self._flat(self.commands, group_idx),
            self._flat(self.actions, group_idx),
            self._flat(self.values, group_idx),
            self._flat(self.advantages, group_idx),
            self._flat(self.returns, group_idx),
            self._flat(self.actions_log_prob, group_idx),
            self._flat(self.mu, group_idx),
            self._flat(self.sigma, group_idx),
            self._flat(self.cost_values, group_idx),
            self._flat(self.cost_advantages, group_idx),
            self._flat(self.cost_returns, group_idx),
        )
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                mb = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
                yield tuple(x[mb] for x in arrays)
