import os

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import argparse


parser = argparse.ArgumentParser(description="Train a Student RL agent via distillation.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--teacher_checkpoint", type=str, default="", help="Path to the teacher model checkpoint.")
parser.add_argument('--headless',       action='store_true', default=False)  # enable visualization by default
parser.add_argument('-c', '--cpu',      action='store_true', default=False)  # use cuda by default
args_cli = parser.parse_args()



args_cli.device = "cpu" if args_cli.cpu else "cuda:0"

class TeacherPolicy(nn.Module):
    def __init__(self, actor_obs_dim=52, privileged_obs_dim=2, action_dim=12):
        super().__init__()

        self.actor_network = nn.Sequential(nn.Linear(actor_obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128,action_dim))
        self.critic_network = nn.Sequential(nn.Linear(actor_obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128,1))
        self.actor_std = nn.Parameter(torch.ones(1, action_dim))

    @torch.no_grad()
    def forward(self, actor_obs, deterministic=True):
        """Gets a deterministic action for inference."""
        action_mean = self.actor_network(actor_obs)
        if deterministic:
            return action_mean
        
        dist = Normal(action_mean, self.actor_std)
        return dist.sample()

def play(args):

    env_cfg= task_registry.get_cfgs(name=args_cli.task, seed=42)
    device = args_cli.device

    env_cfg.viewer.sync_frame_time = True

    if(args_cli.num_envs):
        env_cfg.env.num_envs = args_cli.num_envs


    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    if env_cfg.terrain.mesh_type == "plane":
        for i in range(2):
            env_cfg.viewer.pos[i] = env_cfg.viewer.pos[i] - env_cfg.terrain.plane_length / 4
            env_cfg.viewer.lookat[i] = env_cfg.viewer.lookat[i] - env_cfg.terrain.plane_length / 4

    # prepare environment
    env = task_registry.make_env(name=args_cli.task, args=args_cli, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    actor_obs_dim = env_cfg.env.num_observations; critic_obs_dim = env_cfg.env.num_observations; action_dim = env_cfg.env.num_actions
    teacher_policy = TeacherPolicy(actor_obs_dim, critic_obs_dim - actor_obs_dim, action_dim).to(device)
        
    try:
        print(f"Loading teacher checkpoint from: {args_cli.teacher_checkpoint}")
        teacher_policy.load_state_dict(torch.load(args_cli.teacher_checkpoint, map_location=device)['policy_state_dict'])
        teacher_policy.eval() # Set to evaluation mode

    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}. Running without models.")
    


    
    flip_flag = True
    import time
    start_time = time.time()
    control_dt = 0.02


    for i in range(20*int(env.max_episode_length)):


        if(i % 50 ==0):
            flip_flag = not flip_flag

        if(flip_flag):
            obs[:,-4:-2] = torch.tensor([0.0,0.6])
        else:
            obs[:,-4:-2] = torch.tensor([0.6,0.0])




        actions = teacher_policy(obs.detach())

        obs, _, rews, dones, infos = env.step(actions.detach())





        elapsed = time.time() - start_time
        if elapsed < control_dt:
            time.sleep(control_dt - elapsed)
        

        

if __name__ == '__main__':
    args = args_cli
    play(args)
