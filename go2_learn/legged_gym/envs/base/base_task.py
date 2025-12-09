import sys
import numpy as np
import torch
import time
from legged_gym.simulator import IsaacGymSimulator

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, sim_device, headless):
        
        self.render_fps = 50
        self.last_frame_time = 0

        self.device = sim_device
        self.headless = headless

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_observations = cfg.env.num_privileged_observations
        self.num_actions = cfg.env.num_actions
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.int)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        if self.num_privileged_observations is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_observations, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None

        self.extras = {}

        self.simulator = IsaacGymSimulator(cfg, sim_params, sim_device, self.headless)

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError