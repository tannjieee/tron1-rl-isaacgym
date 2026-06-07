import torch
from isaacgym.torch_utils import quat_mul, quat_rotate_inverse

from legged_gym.envs.wheelfoot_flat.wheelfoot_flat import BipedWF as BaseBipedWF


class BipedWF(BaseBipedWF):
    """Wheelfoot env with prepared clean/noisy history views for future VICReg use.

    This class does not add any VICReg loss and does not change the actor/critic
    interface. It only prepares the data tensors needed by a future contrastive
    encoder objective:

    - obs_history: noisy policy history, unchanged public interface
    - obs_history_clean: same 10-frame window before additive sensor noise
    - vicreg_obs_history_view1/view2: two independently noised views of the
      same clean history window

    Previous actions keep zero noise because they are control records rather than
    measured sensor channels.
    """

    def _init_buffers(self):
        super()._init_buffers()
        self.obs_buf_clean = torch.zeros_like(self.obs_buf)
        self.obs_history_clean = torch.zeros_like(self.obs_history)
        self.vicreg_obs_history_view1 = torch.zeros_like(self.obs_history)
        self.vicreg_obs_history_view2 = torch.zeros_like(self.obs_history)

    def _apply_randomized_imu_offset_to_obs(self, obs_buf):
        if self.cfg.domain_rand.randomize_imu_offset:
            randomized_base_quat = quat_mul(self.random_imu_offset, self.base_quat)
            obs_buf[:, :3] = (
                quat_rotate_inverse(randomized_base_quat, self.root_states[:, 10:13])
                * self.obs_scales.ang_vel
            )
            obs_buf[:, 3:6] = quat_rotate_inverse(
                randomized_base_quat, self.gravity_vec
            )
        return obs_buf

    def _add_independent_obs_noise(self, obs_buf):
        if not self.add_noise:
            return obs_buf.clone()
        noise = (2.0 * torch.rand_like(obs_buf) - 1.0) * self.noise_scale_vec
        return obs_buf + noise

    def _make_vicreg_history_views(self, clean_history):
        """Build two positive views from the same clean history window.

        The two views use identical time windows and independent observation
        noise. No temporal shift and no feature dropout are applied here.
        """
        if not self.add_noise:
            return clean_history.clone(), clean_history.clone()

        hist = clean_history.view(self.num_envs, self.obs_history_length, self.num_obs)
        noise_scale = self.noise_scale_vec.view(1, 1, self.num_obs)
        view1 = hist + (2.0 * torch.rand_like(hist) - 1.0) * noise_scale
        view2 = hist + (2.0 * torch.rand_like(hist) - 1.0) * noise_scale
        return (
            view1.reshape(self.num_envs, self.obs_history_length * self.num_obs),
            view2.reshape(self.num_envs, self.obs_history_length * self.num_obs),
        )

    def _replace_actor_slice_in_critic_obs(self, actor_obs_clean):
        """Keep the critic actor-observation slice consistent with IMU-offset obs."""
        start = 3  # base linear velocity occupies the first three critic dims
        end = start + self.num_obs
        if self.critic_obs_buf.shape[1] >= end:
            self.critic_obs_buf[:, start:end] = actor_obs_clean

    def _roll_history_buffers(self, actor_obs_clean, actor_obs_noisy):
        self.obs_history_clean = torch.cat(
            (self.obs_history_clean[:, self.num_obs:], actor_obs_clean), dim=-1
        )
        self.obs_history = torch.cat(
            (self.obs_history[:, self.num_obs:], actor_obs_noisy), dim=-1
        )

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if reset_env_ids.numel() > 0:
            self.obs_history_clean[reset_env_ids] = actor_obs_clean[
                reset_env_ids
            ].repeat(1, self.obs_history_length)
            self.obs_history[reset_env_ids] = actor_obs_noisy[reset_env_ids].repeat(
                1, self.obs_history_length
            )

    def compute_observations(self):
        """Compute actor/critic observations and prepare VICReg input views.

        The ordering is intentionally:
        raw proprioception -> IMU offset -> clean history -> independent policy
        noise -> noisy policy history -> two independent VICReg views.
        """
        actor_obs_raw, self.critic_obs_buf = self.compute_group_observations()

        actor_obs_clean = self._apply_randomized_imu_offset_to_obs(actor_obs_raw.clone())
        self.obs_buf_clean = actor_obs_clean
        self._replace_actor_slice_in_critic_obs(actor_obs_clean)

        actor_obs_noisy = self._add_independent_obs_noise(actor_obs_clean)
        self.obs_buf = actor_obs_noisy

        self._roll_history_buffers(actor_obs_clean, actor_obs_noisy)
        (
            self.vicreg_obs_history_view1,
            self.vicreg_obs_history_view2,
        ) = self._make_vicreg_history_views(self.obs_history_clean)

    def get_vicreg_observations(self):
        """Return prepared inputs for future VICReg-style encoder training."""
        return (
            self.obs_history_clean,
            self.vicreg_obs_history_view1,
            self.vicreg_obs_history_view2,
        )
