import math
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import random

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.utils.helpers import class_to_dict

class BipedWF(BaseTask):
    def __init__(
        self, cfg: BaseConfig, sim_params, physics_engine, sim_device, headless
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2
        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self._prepare_cost_function()
        self.init_done = True

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        # self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        # self.obs_history[env_ids] = 0
        # obs_buf, _ = self.compute_group_observations()
        # self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.obs_history[env_ids] = 0
        self.obs_history_clean[env_ids] = 0
        self.vicreg_obs_history_view1[env_ids] = 0
        self.vicreg_obs_history_view2[env_ids] = 0
        obs_buf, _ = self.compute_group_observations()
        obs_buf = self._apply_randomized_imu_offset_to_obs(obs_buf.clone())
        obs_buf_noisy = self._add_independent_obs_noise(obs_buf)
        self.obs_history_clean[env_ids] = obs_buf[env_ids].repeat(
            1, self.obs_history_length
        )
        self.obs_history[env_ids] = obs_buf_noisy[env_ids].repeat(
            1, self.obs_history_length
        )
        (
            self.vicreg_obs_history_view1,
            self.vicreg_obs_history_view2,
        ) = self._make_vicreg_history_views(self.obs_history_clean)
        
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        if hasattr(self, "kinematic_utility_target"):
            self.kinematic_utility_target[env_ids] = 0.0
            self.last_kinematic_utility[env_ids] = 1.0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        for key in self.cost_episode_sums.keys():
            self.extras["episode"]["cost_" + key] = (
                torch.mean(self.cost_episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.cost_episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def step(self, actions):
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )
        
    def _action_clip(self, actions):
        self.actions = actions
        
    def _compute_torques(self, actions):
        pos_action = (
            torch.cat(
                (
                    actions[:, 0:3], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                    actions[:, 4:7], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_pos
        )
        vel_action = (
            torch.cat(
                (
                    torch.zeros_like(actions[:, 0:3]), actions[:, 3].view(self.num_envs, 1),
                    torch.zeros_like(actions[:, 0:3]), actions[:, 7].view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_vel
        )
        # pd controller
        torques = self.p_gains * (pos_action + self.default_dof_pos - self.dof_pos) + self.d_gains * (vel_action - self.dof_vel)
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits )
        return torques * self.torques_scale

    def post_physics_step(self):
        super().post_physics_step()
        self.wheel_lin_vel = self.foot_velocities[:, 0, :] + self.foot_velocities[:, 1, :]
        self._update_kinematic_utility_target()

    def _get_ku_cfg_value(self, name, default):
        ku_cfg = getattr(self.cfg, "kinematic_utility", None)
        return getattr(ku_cfg, name, default) if ku_cfg is not None else default

    def _is_left_dof(self, name):
        return "_L_" in name or name.endswith("_L_Joint") or "_L" in name

    def _is_right_dof(self, name):
        return "_R_" in name or name.endswith("_R_Joint") or "_R" in name

    def _dof_indices(self, side="left", include_wheel=False):
        indices = []
        for i, name in enumerate(self.dof_names):
            lname = name.lower()
            is_side = self._is_left_dof(name) if side == "left" else self._is_right_dof(name)
            if not is_side:
                continue
            is_wheel = "wheel" in lname
            if include_wheel == is_wheel:
                indices.append(i)
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _safe_mean_or_one(self, values):
        if values.numel() == 0:
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        return values.mean(dim=-1)

    def _feet_pos_base_frame(self):
        feet_pos_base = self.foot_positions - self.base_position.unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            feet_pos_base[:, i, :] = quat_rotate_inverse(self.base_quat, feet_pos_base[:, i, :])
        return feet_pos_base

    def _feet_vel_base_frame(self):
        feet_vel_base = self.foot_velocities.clone()
        for i in range(len(self.feet_indices)):
            feet_vel_base[:, i, :] = quat_rotate_inverse(self.base_quat, feet_vel_base[:, i, :])
        return feet_vel_base

    def _compute_leg_joint_margin_utility(self, dof_ids):
        if dof_ids.numel() == 0:
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        q = self.dof_pos[:, dof_ids]
        lower = self.dof_pos_limits[dof_ids, 0].unsqueeze(0)
        upper = self.dof_pos_limits[dof_ids, 1].unsqueeze(0)
        half_range = torch.clamp(0.5 * (upper - lower), min=1.0e-6)
        normalized_margin = torch.minimum(q - lower, upper - q) / half_range
        margin_floor = float(self._get_ku_cfg_value("joint_margin_floor", 0.08))
        margin_temp = float(self._get_ku_cfg_value("joint_margin_temp", 0.04))
        utility_per_joint = torch.sigmoid((normalized_margin - margin_floor) / margin_temp)
        return torch.clamp(utility_per_joint.min(dim=-1).values, 0.0, 1.0)

    def _compute_workspace_utility(self, side_index):
        feet_pos_base = self._feet_pos_base_frame()
        foot_pos = feet_pos_base[:, side_index, :]
        nominal = self.ku_nominal_foot_pos_base[:, side_index, :]
        x_scale = float(self._get_ku_cfg_value("workspace_x_scale", 0.35))
        y_scale = float(self._get_ku_cfg_value("workspace_y_scale", 0.12))
        z_scale = float(self._get_ku_cfg_value("workspace_z_scale", 0.18))
        err = torch.stack(
            (
                (foot_pos[:, 0] - nominal[:, 0]) / x_scale,
                (foot_pos[:, 1] - nominal[:, 1]) / y_scale,
                (foot_pos[:, 2] - nominal[:, 2]) / z_scale,
            ),
            dim=-1,
        )
        return torch.exp(-torch.sum(torch.square(err), dim=-1)).clamp(0.0, 1.0)

    def _compute_roll_consistency_utility(self, side_index, wheel_dof_ids):
        if wheel_dof_ids.numel() == 0:
            return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        feet_vel_base = self._feet_vel_base_frame()
        wheel_center_vx = feet_vel_base[:, side_index, 0]
        wheel_speed = self.dof_vel[:, wheel_dof_ids[0]] * self.cfg.asset.foot_radius
        # The wheel joint sign can differ between URDF conventions, so use the smaller
        # of the two signed rolling residuals as a sign-invariant slip proxy.
        residual_same = torch.abs(wheel_center_vx - wheel_speed)
        residual_flip = torch.abs(wheel_center_vx + wheel_speed)
        residual = torch.minimum(residual_same, residual_flip)
        sigma = float(self._get_ku_cfg_value("roll_consistency_sigma", 0.35))
        return torch.exp(-torch.square(residual / sigma)).clamp(0.0, 1.0)

    def _update_kinematic_utility_target(self):
        if not hasattr(self, "kinematic_utility_target"):
            return
        left_joint_u = self._compute_leg_joint_margin_utility(self.left_leg_dof_ids)
        right_joint_u = self._compute_leg_joint_margin_utility(self.right_leg_dof_ids)
        left_workspace_u = self._compute_workspace_utility(0)
        right_workspace_u = self._compute_workspace_utility(1)
        left_roll_u = self._compute_roll_consistency_utility(0, self.left_wheel_dof_ids)
        right_roll_u = self._compute_roll_consistency_utility(1, self.right_wheel_dof_ids)

        left_k = (left_joint_u * left_workspace_u * left_roll_u).clamp(0.0, 1.0)
        right_k = (right_joint_u * right_workspace_u * right_roll_u).clamp(0.0, 1.0)
        k_pair = torch.stack((left_k, right_k), dim=-1)
        dk_pair = (k_pair - self.last_kinematic_utility) / max(self.dt, 1.0e-6)
        dk_scale = float(self._get_ku_cfg_value("dku_scale", 5.0))
        dk_pair = torch.clamp(dk_pair / dk_scale, -1.0, 1.0)

        self.kinematic_utility_target = torch.cat(
            (
                k_pair,
                torch.stack((left_joint_u, right_joint_u), dim=-1),
                torch.stack((left_workspace_u, right_workspace_u), dim=-1),
                torch.stack((left_roll_u, right_roll_u), dim=-1),
                (left_k - right_k).unsqueeze(-1),
                dk_pair,
            ),
            dim=-1,
        )
        self.last_kinematic_utility = k_pair.detach()
        self.extras["ku_target"] = self.kinematic_utility_target.detach()
        self.extras["ku_mean"] = k_pair.mean(dim=0).detach()

    def _build_nominal_gains(self, gain_cfg):
        nominal_gains = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            for dof_name, gain in gain_cfg.items():
                if dof_name in name:
                    nominal_gains[:, i] = gain
                    break
        return nominal_gains

    def _safe_scale(self, value, nominal_value):
        return torch.where(
            torch.abs(nominal_value) > 1e-6,
            value / nominal_value,
            torch.ones_like(value),
        )

    def _privileged_attr(self, attr_name, width, default_value=0.0):
        value = getattr(self, attr_name, None)
        if value is None:
            return torch.full(
                (self.num_envs, width),
                default_value,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=torch.float, device=self.device)
        else:
            value = value.to(device=self.device, dtype=torch.float)
        if value.dim() == 0:
            value = value.view(1, 1).repeat(self.num_envs, width)
        elif value.dim() == 1:
            value = value.unsqueeze(-1)
        value = value.view(self.num_envs, -1)
        if value.shape[1] != width:
            value = value[:, :width]
        return value

    def _get_privileged_domain_rand_obs(self):
        friction = self._privileged_attr("friction_coeffs", 1)
        restitution = self._privileged_attr("restitution_coef", 1)
        base_mass = self._privileged_attr("base_mass", 1)
        base_com = self._privileged_attr("base_com", 3)
        inertia_scale = self._privileged_attr("inertia_scale", 1)
        action_delay = self.action_delay_idx.float().unsqueeze(-1) * self.sim_params.dt
        imu_offset = self._privileged_attr("random_imu_offset", 4, default_value=0.0)

        p_gain_scale = self._safe_scale(self.p_gains, self.nominal_p_gains)
        d_gain_scale = self._safe_scale(self.d_gains, self.nominal_d_gains)
        motor_torque_scale = self.torques_scale
        default_dof_pos_offset = self.default_dof_pos - self.raw_default_dof_pos.unsqueeze(0)

        push_force = self.rigid_body_external_forces[:, 0, :]
        if self.cfg.domain_rand.push_robots:
            max_push_force = (
                self.base_mass.mean().clamp_min(1e-6)
                * self.cfg.domain_rand.max_push_vel_xy
                / self.sim_params.dt
            )
            push_force = push_force / max_push_force

        return torch.cat(
            (
                friction,
                restitution,
                base_mass,
                base_com,
                inertia_scale,
                action_delay,
                imu_offset,
                p_gain_scale,
                d_gain_scale,
                motor_torque_scale,
                default_dof_pos_offset,
                push_force,
            ),
            dim=-1,
        )

    def _prepare_cost_function(self):
        costs_cfg = getattr(self.cfg, "costs", None)
        if costs_cfg is None:
            self.cost_scales = {}
            self.cost_d_values = {}
            self.cost_names = []
            self.cost_functions = []
            self.num_costs = 0
            self.cost_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.cost_terms_buf = torch.zeros(self.num_envs, 0, dtype=torch.float, device=self.device)
            self.cost_episode_sums = {}
            self.cost_k_values = torch.zeros(1, 0, dtype=torch.float, device=self.device)
            self.cost_d_values_tensor = torch.zeros(1, 1, 0, dtype=torch.float, device=self.device)
            return

        self.cost_scales = class_to_dict(getattr(costs_cfg, "scales", {}))
        self.cost_d_values = class_to_dict(getattr(costs_cfg, "d_values", {}))
        self.cost_names = []
        self.cost_functions = []
        self.cost_k_values = []
        self.cost_d_values_tensor = []
        for name, scale in list(self.cost_scales.items()):
            if scale == 0:
                self.cost_scales.pop(name)
                continue
            function_name = "_cost_" + name
            if not hasattr(self, function_name):
                raise AttributeError(f"Cost '{name}' is configured but {function_name} is not implemented")
            self.cost_names.append(name)
            self.cost_functions.append(getattr(self, function_name))
            self.cost_k_values.append(float(scale))
            self.cost_d_values_tensor.append(float(self.cost_d_values.get(name, 0.0)))

        self.num_costs = len(self.cost_names)
        self.cost_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.cost_terms_buf = torch.zeros(
            self.num_envs,
            self.num_costs,
            dtype=torch.float,
            device=self.device,
        )
        self.cost_episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.cost_names
        }
        self.cost_k_values = torch.tensor(
            self.cost_k_values,
            dtype=torch.float,
            device=self.device,
        ).view(1, -1)
        self.cost_d_values_tensor = torch.tensor(
            self.cost_d_values_tensor,
            dtype=torch.float,
            device=self.device,
        ).view(1, 1, -1)

    def compute_cost(self):
        self.cost_buf[:] = 0.0
        if len(self.cost_functions) == 0:
            return

        use_dt = bool(getattr(self.cfg.costs, "use_dt", True))
        dt_scale = self.dt if use_dt else 1.0
        for i, function in enumerate(self.cost_functions):
            name = self.cost_names[i]
            cost = function() * dt_scale
            self.cost_terms_buf[:, i] = cost
            self.cost_buf += cost
            self.cost_episode_sums[name] += cost

        self.extras["cost"] = self.cost_buf.detach()
        self.extras["cost_terms"] = self.cost_terms_buf.detach()

    def _safe_limit(self, value):
        return torch.clamp(torch.abs(value), min=1.0e-6)

    def _wheel_dof_mask(self):
        name_keys = getattr(self.cfg.costs, "wheel_name_keys", ["wheel"])
        return torch.tensor(
            [
                any(key.lower() in name.lower() for key in name_keys)
                for name in self.dof_names
            ],
            dtype=torch.bool,
            device=self.device,
        )

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        dof_list = [0,1,2,4,5,6]
        dof_pos = (self.dof_pos - self.default_dof_pos)[:,dof_list]
        # dof_pos = torch.remainder(dof_pos + self.pi, 2 * self.pi) - self.pi

        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                dof_pos * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                # self.clock_inputs_sin.view(self.num_envs, 1),
                # self.clock_inputs_cos.view(self.num_envs, 1),
                # self.gaits,
            ),
            dim=-1,
        )
        critic_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                obs_buf,
                self._get_privileged_domain_rand_obs(),
            ),
            dim=-1,
        )
        return obs_buf, critic_obs_buf

    def compute_observations(self):
        """
        Computes actor/critic observations and prepares clean/noisy history
        tensors for future VICReg-style encoder training.

        No VICReg loss is added here.
        No feature dropout is used.
        """
        actor_obs_raw, self.critic_obs_buf = self.compute_group_observations()

        actor_obs_clean = self._apply_randomized_imu_offset_to_obs(
            actor_obs_raw.clone()
        )
        self.obs_buf_clean = actor_obs_clean
        self._replace_actor_slice_in_critic_obs(actor_obs_clean)

        actor_obs_noisy = self._add_independent_obs_noise(actor_obs_clean)
        self.obs_buf = actor_obs_noisy

        self._roll_history_buffers(actor_obs_clean, actor_obs_noisy)

        (
            self.vicreg_obs_history_view1,
            self.vicreg_obs_history_view2,
        ) = self._make_vicreg_history_views(self.obs_history_clean)

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            if env_id == 0:
                min_add_mass, max_add_mass = self.cfg.domain_rand.added_mass_range
                self.base_add_mass = (
                    torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)\
                    * (max_add_mass - min_add_mass) + min_add_mass)
                self.base_mass = props[0].mass + self.base_add_mass
            props[0].mass += self.base_add_mass[env_id]
        else:
            self.base_mass[:] = props[0].mass

        if self.cfg.domain_rand.randomize_base_com:
            if env_id == 0:
                com_x, com_y, com_z = self.cfg.domain_rand.rand_com_vec
                self.base_com[:, 0] = (
                    torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)\
                    * (com_x * 2) - com_x)
                self.base_com[:, 1] = (
                    torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)\
                    * (com_y * 2) - com_y)
                self.base_com[:, 2] = (
                    torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)\
                    * (com_z * 2) - com_z)
            props[0].com.x += self.base_com[env_id, 0]
            props[0].com.y += self.base_com[env_id, 1]
            props[0].com.z += self.base_com[env_id, 2]

        if env_id == 0:
            self.inertia_scale = torch.ones(
                self.num_envs,
                1,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        if self.cfg.domain_rand.randomize_inertia:
            low_bound, high_bound = self.cfg.domain_rand.randomize_inertia_range
            inertia_scales = []
            for i in range(len(props)):
                inertia_scale = np.random.uniform(low_bound, high_bound)
                inertia_scales.append(inertia_scale)
                props[i].mass *= inertia_scale
                props[i].inertia.x.x *= inertia_scale
                props[i].inertia.y.y *= inertia_scale
                props[i].inertia.z.z *= inertia_scale
            self.inertia_scale[env_id, 0] = float(np.mean(inertia_scales))
        return props
     
    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        # self._resample_gaits(env_ids)
        # self._step_contact_targets()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 0.1 * wrap_to_pi(self.commands[:, 3] - heading)

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        #set 50% of resample to go straight
        resample_nums = len(env_ids)
        env_list = list(range(resample_nums))
        half_env_list = random.sample(env_list, resample_nums // 2)
        # forward = quat_apply(self.base_quat[env_ids[half_env_list]], \
        #                      self.forward_vec[env_ids[half_env_list]])
        # heading = torch.atan2(forward[:,1], forward[:,0])
        # self.commands[env_ids[half_env_list], 3] = heading
        
        # set 20% of the rest 50% to be stand still
        rest_env_list = list(set(env_list) - set(half_env_list))
        zero_cmd_env_idx_ = random.sample(rest_env_list, resample_nums // 2 // 5)

        self.commands[env_ids[zero_cmd_env_idx_], 0] = 0.0
        self.commands[env_ids[zero_cmd_env_idx_], 1] = 0.0
        self.commands[env_ids[zero_cmd_env_idx_], 2] = 0.0
        #use heading
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat[env_ids[zero_cmd_env_idx_]], \
                                 self.forward_vec[env_ids[zero_cmd_env_idx_]])
            heading = torch.atan2(forward[:,1], forward[:,0])
            self.commands[env_ids[zero_cmd_env_idx_], 3] = heading
            
    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12:20] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[20:] = 0.0  # previous actions
        return noise_vec

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
        if not self.add_noise:
            return clean_history.clone(), clean_history.clone()

        hist = clean_history.view(
            self.num_envs,
            self.obs_history_length,
            self.num_obs,
        )
        noise_scale = self.noise_scale_vec.view(1, 1, self.num_obs)

        view1 = hist + (2.0 * torch.rand_like(hist) - 1.0) * noise_scale
        view2 = hist + (2.0 * torch.rand_like(hist) - 1.0) * noise_scale

        return (
            view1.reshape(self.num_envs, self.obs_history_length * self.num_obs),
            view2.reshape(self.num_envs, self.obs_history_length * self.num_obs),
        )

    def _replace_actor_slice_in_critic_obs(self, actor_obs_clean):
        start = 3  # base_lin_vel occupies critic_obs[:, 0:3]
        end = start + self.num_obs
        if self.critic_obs_buf.shape[1] >= end:
            self.critic_obs_buf[:, start:end] = actor_obs_clean

    def _roll_history_buffers(self, actor_obs_clean, actor_obs_noisy):
        self.obs_history_clean = torch.cat(
            (self.obs_history_clean[:, self.num_obs:], actor_obs_clean),
            dim=-1,
        )
        self.obs_history = torch.cat(
            (self.obs_history[:, self.num_obs:], actor_obs_noisy),
            dim=-1,
        )

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if reset_env_ids.numel() > 0:
            self.obs_history_clean[reset_env_ids] = actor_obs_clean[
                reset_env_ids
            ].repeat(1, self.obs_history_length)
            self.obs_history[reset_env_ids] = actor_obs_noisy[
                reset_env_ids
            ].repeat(1, self.obs_history_length)

    def get_vicreg_observations(self):
        return (
            self.obs_history_clean,
            self.vicreg_obs_history_view1,
            self.vicreg_obs_history_view2,
        )

    def _init_buffers(self):
        super()._init_buffers()
        self.nominal_p_gains = self._build_nominal_gains(self.cfg.control.stiffness)
        self.nominal_d_gains = self._build_nominal_gains(self.cfg.control.damping)
        self.wheel_lin_vel = torch.zeros_like(self.foot_velocities)
        self.wheel_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.left_leg_dof_ids = self._dof_indices(side="left", include_wheel=False)
        self.right_leg_dof_ids = self._dof_indices(side="right", include_wheel=False)
        self.left_wheel_dof_ids = self._dof_indices(side="left", include_wheel=True)
        self.right_wheel_dof_ids = self._dof_indices(side="right", include_wheel=True)

        self.ku_target_dim = int(self._get_ku_cfg_value("target_dim", 11))
        self.kinematic_utility_target = torch.zeros(
            self.num_envs,
            self.ku_target_dim,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_kinematic_utility = torch.ones(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.ku_nominal_foot_pos_base = self._feet_pos_base_frame().detach().clone()
        # buffers for encoder observation history and VICReg
        self.obs_buf_clean = torch.zeros_like(self.obs_buf)
        self.obs_history_clean = torch.zeros_like(self.obs_history)
        self.vicreg_obs_history_view1 = torch.zeros_like(self.obs_history)
        self.vicreg_obs_history_view2 = torch.zeros_like(self.obs_history)


    # ------------ reward functions----------------

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_nominal_foot_position(self):
        #1. calculate foot postion wrt base in base frame  
        nominal_base_height = -(self.cfg.rewards.base_height_target- self.cfg.asset.foot_radius)
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        reward = 0
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
            height_error = nominal_base_height - foot_positions_base[:, i, 2]
            reward += torch.exp(-(height_error ** 2)/ self.cfg.rewards.nominal_foot_position_tracking_sigma)
        vel_cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        return reward / len(self.feet_indices)*torch.exp(-(vel_cmd_norm ** 2)/self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)
    
    def _reward_same_foot_z_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_z_position_err = foot_positions_base[:,0,2] - foot_positions_base[:,1,2]
        return foot_z_position_err ** 2

    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        return torch.exp(-(leg_symmetry_err ** 2)/ self.cfg.rewards.leg_symmetry_tracking_sigma)

    def _reward_same_foot_x_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        reward = torch.abs(foot_x_position_err)
        return reward

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_lin_vel() - self.rwd_linVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_tracking_ang_vel_pb(self):
        delta_phi = ~self.reset_buf * (self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev)
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.abs(base_height - self.cfg.rewards.base_height_target)

    # ------------ cost functions----------------

    def _cost_dof_pos_limit(self):
        half_range = 0.5 * (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0])
        out_of_limits = torch.clamp(self.dof_pos_limits[:, 0] - self.dof_pos, min=0.0)
        out_of_limits += torch.clamp(self.dof_pos - self.dof_pos_limits[:, 1], min=0.0)
        return torch.sum(out_of_limits / self._safe_limit(half_range), dim=1)

    def _cost_dof_vel_limit(self):
        limit = self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
        excess = torch.clamp(torch.abs(self.dof_vel) - limit, min=0.0)
        return torch.sum(excess / self._safe_limit(self.dof_vel_limits), dim=1)

    def _cost_torque_limit(self):
        limit = self.torque_limits * self.cfg.rewards.soft_torque_limit
        excess = torch.clamp(torch.abs(self.torques) - limit, min=0.0)
        return torch.sum(excess / self._safe_limit(self.torque_limits), dim=1)

    def _cost_wheel_vel_limit(self):
        mask = self._wheel_dof_mask()
        if not torch.any(mask):
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        excess = torch.clamp(torch.abs(self.dof_vel[:, mask]) - self.dof_vel_limits[mask], min=0.0)
        return torch.sum(excess / self._safe_limit(self.dof_vel_limits[mask]), dim=1)

    def _cost_collision(self):
        threshold = self.cfg.costs.contact_force_threshold
        return torch.sum(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > threshold,
            dim=1,
        ).float()

    def _cost_termination_contact(self):
        threshold = self.cfg.costs.termination_contact_force_threshold
        return torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > threshold,
            dim=1,
        ).float()

    def _cost_fall(self):
        return (self.projected_gravity[:, 2] > self.cfg.costs.fall_projected_gravity_z).float()

    def _cost_power_limit(self):
        return self.power_limit_out_buf.float()
