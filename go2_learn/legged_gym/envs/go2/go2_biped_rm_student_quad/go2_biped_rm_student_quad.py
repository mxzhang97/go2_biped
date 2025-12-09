from time import time
import numpy as np
import os


import torch
from torch import nn
from typing import Tuple, Dict

from legged_gym.envs.base.base_task import BaseTask
from legged_gym import LEGGED_GYM_ROOT_DIR
from .go2_biped_rm_student_quad_config import Go2BipedRMStudentQuadCfg
from legged_gym.utils.math_utils import *
from legged_gym.utils.utils import *


torch.autograd.set_detect_anomaly(True)


class Go2BipedRMStudentQuad(BaseTask):
    cfg: Go2BipedRMStudentQuadCfg

    def __init__(self, cfg: Go2BipedRMStudentQuadCfg, sim_params: dict, sim_device: str, headless: bool):

        self.cfg = cfg
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, sim_device, headless)


        self.pen_vec = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                   requires_grad=False)


        self.rm_state = torch.zeros(self.num_envs, device = self.device, dtype=torch.long)
        self.rm_last_transition_time = torch.zeros(self.num_envs, device= self.device, dtype=torch.long)
        self.pen_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.quad_mode = torch.zeros(self.num_envs, dtype = torch.bool, device = self.device)

        self.ground_truth_targets = torch.zeros(self.num_envs,6, device=self.device, dtype = torch.float, requires_grad=False)


        print("COM Offset", self.simulator._base_com_bias)
        print("Mass Offset", self.simulator._mass[:,0])
        print("Friction Offset",self.simulator._friction_values)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(
            actions, -clip_actions, clip_actions).to(self.device)
        self.actions[:] = actions[:]
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[torch.arange(
                self.num_envs), self.action_delay].clone()
        self.simulator.step(actions)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)

        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, self.ground_truth_targets


    def compute_observations(self):

        rigid_body_pos = self.simulator.rigid_body_states.view(self.num_envs, self.simulator.num_bodies, 13)[:,:,0:3]- self.simulator.root_states[:, :3].unsqueeze(1) 
        self.feet_pos = self.simulator.rigid_body_states.view(self.num_envs, self.simulator.num_bodies, 13)[:, self.simulator.feet_indices, 0:3] - self.simulator.root_states[:, :3].unsqueeze(1) 
        self.feet_normal_force = torch.clamp(self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2], min=0.0) + torch.tensor([[0, 0, 1e-6, 1e-6]] * self.num_envs, device=self.device)

        self.cop_pos = (1./torch.sum(self.feet_normal_force, dim=-1)).unsqueeze(-1) * torch.sum(self.feet_pos * self.feet_normal_force.unsqueeze(-1), dim=1)

        rear_feet_pos = self.feet_pos[:, 2:, :]
        self.cop_pos = torch.mean(rear_feet_pos, dim=1)

        self._check_nan(self.cop_pos, name="cop_pos", 
                feet_pos=self.feet_pos, 
                feet_normal_force=self.feet_normal_force)

        self._check_inf(self.cop_pos, name="cop_pos", 
                feet_pos=self.feet_pos, 
                feet_normal_force=self.feet_normal_force)

        base_quat = self.simulator.rigid_body_states.view(self.num_envs, self.simulator.num_bodies, 13)[:, 0, 3:7]
        com_offset = self.simulator._base_com_bias
        rotated_offset = quat_apply(base_quat, com_offset)
        rigid_body_pos_w_offset = rigid_body_pos.clone()
        rigid_body_pos_w_offset[:,0,:] += rotated_offset

        single_robot_mass = torch.sum(self.simulator._mass, dim=1).unsqueeze(-1)
        self.com_pos = torch.sum(rigid_body_pos_w_offset * self.simulator._mass.unsqueeze(-1),dim=1)/(single_robot_mass + 1e-6)

        self._check_nan(self.com_pos, name="com_pos", rigid_body_pos = rigid_body_pos_w_offset, mass = self.simulator._mass.unsqueeze(-1))
        self._check_inf(self.com_pos, name="com_pos", rigid_body_pos = rigid_body_pos_w_offset, mass = self.simulator._mass.unsqueeze(-1))

        self.com_cop = self.com_pos - self.cop_pos
        self.pen_vec = self.com_cop

        rm_state_one_hot = torch.nn.functional.one_hot(self.rm_state, num_classes=2) 

        self.obs_buf = torch.cat((self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                                    self.simulator.base_ang_vel * self.obs_scales.ang_vel,
                                    self.simulator.projected_gravity,
                                    self.pen_vec,
                                    (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.simulator.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.commands[:, :3] * self.commands_scale,
                                    self.commands[:, 3:],
                                    self.quad_mode.float().unsqueeze(-1)
                                    ), dim=-1)

        self._check_nan(self.obs_buf, name = "Obs Tensor")
        self._check_inf(self.obs_buf, name = "Obs Tensor")

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.privileged_obs_buf = rm_state_one_hot
        self.ground_truth_targets = torch.cat((self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                                               self.pen_vec), dim=-1)


    def compute_reward(self):
            self.rew_buf[:] = 0.
            self.pen_buf[:] = 0.
            for i in range(len(self.reward_functions)):
                name = self.reward_names[i]
                rew = self.reward_functions[i]() * self.reward_scales[name]
                """
                if(name in ["action_rate","rear_hip_joint","lin_vel_z"]):
                    self.pen_buf += rew
                else:
                    self.rew_buf += rew
                """

                self.rew_buf += rew

                self.episode_sums[name] += rew

            # add termination reward after clipping
            if "termination" in self.reward_scales:
                rew = self._reward_termination(
                ) * self.reward_scales["termination"]
                self.rew_buf += rew
                self.episode_sums["termination"] += rew

            gait_bonus = self._compute_gait_reward_and_update_rm()
            #print("gait_bonus", gait_bonus)
            #print("self.rm_state", self.rm_state)
            #print("self.rm_last_transition", self.rm_last_transition_time)
            self.episode_sums["gait_bonus"] += torch.where(gait_bonus<1.0, 1.0, 0.0)
            #self.pen_buf *= gait_bonus
            #self.rew_buf += self.pen_buf

            if self.cfg.rewards.only_positive_rewards:
                self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

    def post_physics_step(self):

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.simulator.post_physics_step()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.simulator.last_dof_vel[:] = self.simulator.dof_vel[:]

    def check_termination(self):

        fail_buf = torch.any(
            torch.norm(self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :], dim=-1)
            > 10.0, dim=1)
        fail_buf |= self.simulator.projected_gravity[:, 2] > self.cfg.rewards.max_projected_gravity
        self.fail_buf += fail_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )

    def reset_idx(self, env_ids):

        if len(env_ids) == 0:
            return

        self._resample_commands(env_ids)
        self.simulator.reset_idx(env_ids)
        self._reset_dofs(env_ids)

        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0
        self.actions[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

        
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


        # reset action queue and delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_queue[env_ids] = 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

        self._resample_mode(env_ids)



    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _,gt = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs,gt


    # callbacks

    def _post_physics_step_callback(self):

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        mode_switch_env_ids = (self.episode_length_buf % int(self.cfg.commands.mode_resampling/self.dt) == 0).nonzero(as_tuple = False).flatten()

        self._resample_mode(mode_switch_env_ids)

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self.simulator.push_robots()


    def _parse_cfg(self, cfg):
        self.dt = self.cfg.sim.dt * self.cfg.control.decimation
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)


        self.friction_value_offset = (self.cfg.domain_rand.friction_range[0] + 
                                      self.cfg.domain_rand.friction_range[1]) / 2  # mean value
        self.kp_scale_offset = (self.cfg.domain_rand.kp_range[0] +
                                self.cfg.domain_rand.kp_range[1]) / 2  # mean value
        self.kd_scale_offset = (self.cfg.domain_rand.kd_range[0] +
                                self.cfg.domain_rand.kd_range[1]) / 2  # mean value
        
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _prepare_reward_function(self):

        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale ==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name =="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

        self.episode_sums['gait_bonus'] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _init_buffers(self):

        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.forward_vec[:, 0] = 1.0
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.commands = torch.zeros(
            (self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=torch.float)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, dtype=torch.float,
                                           requires_grad=False)
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # last last actions
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.simulator.feet_indices)), device=self.device, dtype=torch.float)
        self.last_contacts = torch.zeros((self.num_envs, len(self.simulator.feet_indices)), device=self.device, dtype=torch.int)
        

        # randomize action delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)

    def _get_noise_scale_vec(self):

        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = noise_scales.com_cop * noise_level  # com_cop
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.  # previous actions
        noise_vec[48:] = 0. #commands

        return noise_vec

    def _check_nan(self, tensor, name="Unnamed Tensor", **inputs):
        if torch.any(torch.isnan(tensor)):


            nan_env_indices = torch.any(
                torch.isnan(tensor.view(self.num_envs, -1)), dim=1
            ).nonzero(as_tuple=False).flatten()


            print(f"\n\n--- NaN DETECTED in '{name}' ---")
            print(f"--- Environments with NaNs: {nan_env_indices.tolist()} ---")

            # Print the filtered inputs that caused the NaN
            for key, value in inputs.items():
                print(f"\n--- Input '{key}' (filtered to NaN envs) ---")
                # Filter each input tensor to show only the rows for the NaN environments
                print(value[nan_env_indices])

            # Print the problematic part of the tensor itself
            print(f"\n--- Resulting Tensor '{name}' with NaNs (filtered) ---")
            print(tensor[nan_env_indices])

            # Drop into the debugger
            print("\n--- Entering Debugger (pdb) ---")
            print("Type 'p <tensor_name>[nan_env_indices]' to inspect filtered tensors.")
            print("Type 'c' to continue, 'q' to quit.")
            breakpoint()



    def _check_inf(self, tensor, name="Unnamed Tensor", **inputs):
        if torch.any(torch.isinf(tensor)):


            nan_env_indices = torch.any(
                torch.isinf(tensor.view(self.num_envs, -1)), dim=1
            ).nonzero(as_tuple=False).flatten()


            print(f"\n\n--- Inf DETECTED in '{name}' ---")
            print(f"--- Environments with Infss: {nan_env_indices.tolist()} ---")

            for key, value in inputs.items():
                print(f"\n--- Input '{key}' (filtered to Inf envs) ---")
                print(value[nan_env_indices])

            print(f"\n--- Resulting Tensor '{name}' with Infs (filtered) ---")
            print(tensor[nan_env_indices])

            print("\n--- Entering Debugger (pdb) ---")
            print("Type 'p <tensor_name>[nan_env_indices]' to inspect filtered tensors.")
            print("Type 'c' to continue, 'q' to quit.")
            breakpoint()


    def _reset_dofs(self, env_ids):

        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[:, :] = self.simulator.default_dof_pos[:] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_actions), self.device)

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

    def _resample_commands(self, env_ids):

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids),1), self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids),1), self.device).squeeze(1)

        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["fl_arm_track_target"][0], self.command_ranges["fl_arm_track_target"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["fr_arm_track_target"][0], self.command_ranges["fr_arm_track_target"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(
            self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


        self.commands[env_ids, 2] *= (
            self.commands[env_ids, 2] > 0.05) #>0.2


        zero_vel_envs = (torch.rand(len(env_ids), device=self.device) <0.1)

        self.commands[env_ids, :3] = torch.where(zero_vel_envs.unsqueeze(-1), torch.zeros_like(self.commands[env_ids,:3]), self.commands[env_ids,:3])


    def _resample_mode(self, env_ids):


        ### quad mode

        quad_mask = (torch.rand(len(env_ids), device=self.device) < 0.1)
        self.quad_mode[env_ids] = quad_mask 




    def _compute_gait_reward_and_update_rm(self):
        rear_foot_indices = self.simulator.feet_indices[2:]
        foot_heights = self.simulator.rigid_body_states.view(self.num_envs, self.simulator.num_bodies, 13)[:, rear_foot_indices, 2]
        rl_foot_height = foot_heights[:, 0]
        rr_foot_height = foot_heights[:, 1]
        
        contacts = self.simulator.link_contact_forces[:, rear_foot_indices, 2] > 1.0
        PRL, PRR = contacts[:, 0], contacts[:, 1]

        swing_foot_high_for_q0 = (rl_foot_height > self.cfg.rewards.min_foot_height)
        swing_foot_high_for_q1 = (rr_foot_height > self.cfg.rewards.min_foot_height)

        self.rm_last_transition_time += 1
        can_transition = (self.rm_last_transition_time >= self.cfg.rewards.min_gait_freq_step)

        condition_for_q1 = (PRL & ~PRR) & can_transition & swing_foot_high_for_q1
        condition_for_q0 = (~PRL & PRR) & can_transition & swing_foot_high_for_q0

        is_in_state_0 = (self.rm_state == 0) 
        is_in_state_1 = (self.rm_state == 1) 

        made_transition_to_q1 = is_in_state_0 & condition_for_q1
        made_transition_to_q0 = is_in_state_1 & condition_for_q0
        all_transitions = made_transition_to_q1 | made_transition_to_q0

        bonus_factor = torch.ones(self.num_envs, device=self.device)
        apply_bonus_mask = all_transitions 
        bonus_factor[apply_bonus_mask] = self.cfg.rewards.gait_transition_bonus
        
        new_rm_state = self.rm_state.clone()
        new_rm_state[made_transition_to_q1] = 1
        new_rm_state[made_transition_to_q0] = 0
        self.rm_state = new_rm_state
        
        self.rm_last_transition_time[all_transitions] = 0

        return bonus_factor





    # reward functions
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(
            self.simulator.link_contact_forces[:, self.simulator.penalized_contact_indices, :], 
            dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf




    def _reward_lin_vel_z(self):

        return torch.square(self.simulator.base_lin_vel[:, 0]) * (1-self.quad_mode.float())

    def _reward_ang_vel_xy(self):

        return torch.sum(torch.square(self.simulator.base_ang_vel[:, 1:3]), dim=1)* (1-self.quad_mode.float())

    def _reward_orientation(self):

        return torch.sum(torch.square(self.simulator.projected_gravity[:, :2]), dim=1) * torch.clamp(-self.simulator.projected_gravity[:,0], min = 0.0) * (1-self.quad_mode.float())

    def _reward_orientation_3(self):

        return torch.sum(torch.square(self.simulator.projected_gravity[:, 2]).unsqueeze(1), dim=1) * (1-self.quad_mode.float())

    def _reward_fl_contact_force(self):
        foot_forces = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_indices, :], dim=-1) 
        fl_force = foot_forces[:, 0]
        return fl_force * (1-self.quad_mode.float())

    def _reward_fr_contact_force(self):
        foot_forces = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_indices, :], dim=-1)
        fr_force = foot_forces[:, 1]
        return fr_force * (1-self.quad_mode.float())

    def _reward_f_contact_force(self):
        foot_forces = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_indices, :], dim=-1)
        f_force = foot_forces[:, 0] + foot_forces[:, 1]
        return f_force * (1-self.quad_mode.float())



    def _reward_tracking_lin_vel(self):

        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] + self.simulator.base_lin_vel[:, 1:3]), dim=1)

        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma) * (1-self.quad_mode.float())

    def _reward_tracking_ang_vel(self):

        ang_vel_error = torch.square(self.commands[:, 2] - self.simulator.base_ang_vel[:, 0])

        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma) * (1-self.quad_mode.float())


    def _reward_inv_pendulum(self):

        self.pen_len = torch.norm(self.pen_vec, dim=1)

        argument = self.pen_vec[:,2] / (self.pen_len + 1e-6)
    
        self._check_nan(argument, name="acos_argument",
                        pen_vec_z=self.pen_vec[:,2],
                        pen_len=self.pen_len)

        self.pen_ang = torch.acos(argument)

        self._check_nan(self.pen_ang, name="pen_ang", argument=argument)


        return torch.pow(self.pen_ang, 2) * (1-self.quad_mode.float())

    def _reward_inv_pendulum_acc(self):


        self.pen_len = torch.norm(self.pen_vec, dim=1)

        self.pen_ang_acc = (torch.ones(self.num_envs,device=self.device, requires_grad=False) - torch.pow(self.pen_vec[:,2] / (1e-6 * torch.ones(self.num_envs, device=self.device, requires_grad=False) + self.pen_len), 2)) / (1e-6 * torch.ones(self.num_envs, device=self.device, requires_grad=False) + torch.pow(self.pen_len, 2))

        return self.pen_ang_acc  * (1-self.quad_mode.float())


    def _reward_cart_table_len_xy(self):

        self.cart_len_xy = torch.norm(self.pen_vec[:, :2], dim=1) 

        return self.cart_len_xy * (1-self.quad_mode.float())
        

    def _reward_fl_arm_track(self):
        
        all_pos = self.simulator.rigid_body_states.view(self.num_envs, self.simulator.num_bodies, 13)[..., 0:3] - self.simulator.root_states[:, :3].unsqueeze(1) 
        base_pos = all_pos[:,0]
        feet_pos = all_pos[:,self.simulator.feet_indices]
        fl_pos = feet_pos[:,0]

        feet_to_base = fl_pos - base_pos

        orientation_check = (self.simulator.projected_gravity[:,0]<-0.9)


        local_frame_dist = quat_apply_inverse(self.simulator.root_states[:, 3:7], feet_to_base)

        extension_dist = -local_frame_dist[:,2]
        clamped_dist = torch.clamp(extension_dist, min = 0.0) - self.commands[:,3]



        return torch.exp(-torch.square(clamped_dist)/0.05)  * orientation_check  * (1-self.quad_mode.float())



    def _reward_fr_arm_track(self):


        all_pos = self.simulator.rigid_body_states.view(self.num_envs, self.simulator.num_bodies, 13)[..., 0:3] - self.simulator.root_states[:, :3].unsqueeze(1) 
        base_pos = all_pos[:,0]
        feet_pos = all_pos[:,self.simulator.feet_indices]
        fr_pos = feet_pos[:,1]


        feet_to_base = fr_pos - base_pos

        orientation_check = (self.simulator.projected_gravity[:,0]<-0.9)


        local_frame_dist = quat_apply_inverse(self.simulator.root_states[:, 3:7], feet_to_base)

        extension_dist = -local_frame_dist[:,2]
        clamped_dist = torch.clamp(extension_dist, min = 0.0) - self.commands[:,4]


        return torch.exp(-torch.square(clamped_dist)/0.05)  * orientation_check  * (1-self.quad_mode.float())


    def _reward_fl_arm_track_lateral_distance(self):
        
        all_pos = self.simulator.rigid_body_states.view(self.num_envs, self.simulator.num_bodies, 13)[..., 0:3] - self.simulator.root_states[:, :3].unsqueeze(1) 
        base_pos = all_pos[:,0]
        feet_pos = all_pos[:,self.simulator.feet_indices]
        fl_pos = feet_pos[:,0]

        feet_to_base = fl_pos - base_pos

        orientation_check = (self.simulator.projected_gravity[:,0]<-0.9)


        local_frame_dist = quat_apply_inverse(self.simulator.root_states[:, 3:7], feet_to_base)

        lateral_dist = local_frame_dist[:,1]
        vertical_dist = local_frame_dist[:,0]


        return (torch.square(lateral_dist-0.18) + torch.square(vertical_dist - 0.15))  * orientation_check * (1-self.quad_mode.float())



    def _reward_fr_arm_track_lateral_distance(self):


        all_pos = self.simulator.rigid_body_states.view(self.num_envs, self.simulator.num_bodies, 13)[..., 0:3] - self.simulator.root_states[:, :3].unsqueeze(1) 
        base_pos = all_pos[:,0]
        feet_pos = all_pos[:,self.simulator.feet_indices]
        fr_pos = feet_pos[:,1]


        feet_to_base = fr_pos - base_pos


        orientation_check = (self.simulator.projected_gravity[:,0]<-0.9)

        local_frame_dist = quat_apply_inverse(self.simulator.root_states[:, 3:7], feet_to_base)

        lateral_dist = local_frame_dist[:,1]
        vertical_dist = local_frame_dist[:,0]


        return (torch.square(lateral_dist+0.18) + torch.square(vertical_dist -0.15))  * orientation_check * (1-self.quad_mode.float())



    def _reward_rear_hip_joint(self):

        dof_diff = self.simulator.dof_pos[:,[6,9]] - self.simulator.default_dof_pos[:,[6,9]]

        return torch.sum(torch.square(dof_diff), dim=-1) * (1-self.quad_mode.float())

    def _reward_front_hip_joint(self):

        dof_diff = self.simulator.dof_pos[:,[0,3]] - self.simulator.default_dof_pos[:,[0,3]]

        return torch.sum(torch.square(dof_diff), dim=-1) * (1-self.quad_mode.float())



    ###sigh i wasted a day of my life on this oh my god.



    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime * (1-self.quad_mode.float())

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.simulator.base_pos[:, 2].unsqueeze(
            1), dim=1)
        # print(f"base height: {base_height}")
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew * (1-self.quad_mode.float())




    def _reward_orientation_quad(self):

        return torch.sum(torch.square(self.simulator.projected_gravity[:,:2]), dim=-1) * self.quad_mode.float()

    def _reward_ang_vel_quad(self):

        #may need to rotate to body frame, but should be fine

        return torch.sum(torch.square(self.simulator.base_ang_vel), dim=-1)* self.quad_mode.float()

    def _reward_lin_vel_quad(self):


        return torch.sum(torch.square(self.simulator.base_lin_vel), dim=-1)* self.quad_mode.float()

    def _reward_torque_quad(self):

        return torch.sum(torch.square(self.simulator.torques), dim=-1)* self.quad_mode.float()

    def _reward_default_pos_quad(self):

        return torch.exp(-torch.sum(torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos), dim=-1))* self.quad_mode.float() 


    def _reward_joint_vel_quad(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)* self.quad_mode.float() 
    

    def _reward_joint_acc_quad(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.simulator.last_dof_vel - 
                                       self.simulator.dof_vel) / self.dt), dim=1)* self.quad_mode.float() 


    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost #* (1-self.quad_mode.float())


    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1) * (1-self.quad_mode.float())
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.simulator.last_dof_vel - 
                                       self.simulator.dof_vel) / self.dt), dim=1) * (1-self.quad_mode.float())


    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)  #* (1-self.quad_mode.float())


