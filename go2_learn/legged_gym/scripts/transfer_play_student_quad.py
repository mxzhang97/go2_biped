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
parser.add_argument("--student_checkpoint", type=str, help = "Path to student checkpoint.")
parser.add_argument('--headless',       action='store_true', default=False)  # enable visualization by default
parser.add_argument('-c', '--cpu',      action='store_true', default=False)  # use cuda by default
args_cli = parser.parse_args()



args_cli.device = "cpu" if args_cli.cpu else "cuda:0"

class StudentPolicy(nn.Module):
    def __init__(self, actor_obs_dim=52, privileged_obs_dim=2, action_dim=12):
        super().__init__()


        self.actor_network = nn.Sequential(nn.Linear(actor_obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128,action_dim))
        self.critic_network = nn.Sequential(nn.Linear(actor_obs_dim + privileged_obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128,1))


        self.actor_network_quad = nn.Sequential(nn.Linear(actor_obs_dim-6, 128), nn.ELU(), nn.Linear(128,128), nn.ELU(), nn.Linear(128,action_dim))
        self.critic_network_quad = nn.Sequential(nn.Linear(actor_obs_dim-6, 128), nn.ELU(), nn.Linear(128,128), nn.ELU(), nn.Linear(128,1))


        self.actor_std = nn.Parameter(torch.ones(1, action_dim))
        self.actor_quad_std = nn.Parameter(torch.ones(1, action_dim))

    @torch.no_grad()
    def forward(self, actor_obs):
        """Gets a deterministic action for inference."""
        action_biped = self.actor_network(actor_obs)

        quad_mask = actor_obs[:,-1].unsqueeze(-1)

        action_mean =  quad_mask  * self.actor_network_quad(actor_obs[:,:-6]) + (1-quad_mask) * action_biped 

        
        return action_mean


class StateEstimator(nn.Module):

    def __init__(self, proprio_dim=42*5, target_dim=6):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(proprio_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, target_dim))


    @torch.no_grad()
    def forward(self, propio_history):

        pred = self.network(propio_history)

        return pred

from collections import deque
        


def play(args):

    env_cfg = task_registry.get_cfgs(name=args_cli.task, seed = 42)
    device = args_cli.device
    # override some parameters for testing
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
    obs, priv_obs, gt = env.reset()

    # load policy
    actor_obs_dim = env_cfg.env.num_observations; critic_obs_dim = env_cfg.env.num_privileged_observations; action_dim = env_cfg.env.num_actions
    student_policy = StudentPolicy(actor_obs_dim, critic_obs_dim , action_dim).to(device)
    state_estimator = StateEstimator(42*5,6).to(device)
        
    try:
        from pathlib import Path

        print(f"Loading student checkpoint from: {args_cli.student_checkpoint}")
        student_policy.load_state_dict(torch.load(args_cli.student_checkpoint, map_location=device)['policy_state_dict'])
        student_policy.eval() # Set to evaluation mode


        se_path = Path(args_cli.student_checkpoint).parent.parent / "se" / "last.pt"

        state_estimator.load_state_dict(torch.load(se_path, map_location = device)['policy'])
        state_estimator.eval()

        if(EXPORT):

            print("Scripting StudentPolicy...")

            CPU = True

            if(CPU):
                student_policy.to('cpu')
                state_estimator.to('cpu')



            scripted_student_policy = torch.jit.script(student_policy)
            
            print("Scripting StateEstimator...")
            scripted_state_estimator = torch.jit.script(state_estimator)

            scripted_student_policy.save("scripted_student_policy_cpu_smooth.pt")
            scripted_state_estimator.save("scripted_state_estimator_cpu_smooth.pt")
            
            print("\nSuccessfully exported scripted models:")
            print(f"  - Scripted Student Policy saved to: scripted_student_policy.pt")
            print(f"  - Scripted State Estimator saved to: scripted_state_estimator.pt")

    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}. Running without models.")
    
    HISTORY_LEN = 5
    PROPRIO_DIM = 42
            
    proprio_history = deque(maxlen=HISTORY_LEN)

    for _ in range(HISTORY_LEN):
        proprio_history.append(torch.zeros(env_cfg.env.num_envs, PROPRIO_DIM, device=device))

    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    
    flip_flag = True
    import time
    start_time = time.time()
    control_dt = 0.02

    quad_mode_flag = False


    for i in range(20*int(env.max_episode_length)):

        """
        if(i % 50 ==0):
            flip_flag = not flip_flag

        if(flip_flag):
            obs[:,-6:-1] = torch.tensor([0.0,0.0,0.0,0.6,0.2])
        else:
            obs[:,-6:-1] = torch.tensor([0.0,0.0,0.0,0.2,0.6])



        """

        if(i % 50 ==0):
            flip_flag = not flip_flag

        if(flip_flag):
            obs[:,-3:-1] = torch.tensor([0.1,0.6])
        else:
            obs[:,-3:-1] = torch.tensor([0.6,0.1])
        
        
        """


        if(i%400 ==0):
            quad_mode_flag = not quad_mode_flag

        if(quad_mode_flag):
            obs[:,-1] = torch.tensor([1.0])

        """


        imu_data = obs[:, 3:9]
        joint_and_action_data = obs[:, 12:48]
        current_proprio = torch.cat((imu_data, joint_and_action_data), dim=-1)

        proprio_history.append(current_proprio)


        history_tensor = torch.cat(list(proprio_history), dim=-1)

        with torch.no_grad(): # Ensure no gradients are computed
            estimated_targets = state_estimator(history_tensor)

        ground_truth_lin_vel = gt[:, 0:3]
        ground_truth_com_cop = gt[:, 3:6]


        est_lin_vel = estimated_targets[0, :3].cpu().numpy()
        gt_lin_vel = ground_truth_lin_vel[0].cpu().numpy()
        est_com_cop = estimated_targets[0, 3:].cpu().numpy()
        gt_com_cop = ground_truth_com_cop[0].cpu().numpy()

        if(i%500 == 0):
            print(f"--- Step {i} ---")
            print("observation", obs[0])
            print(f"Lin Vel | GT: [{gt_lin_vel[0]:.3f}, {gt_lin_vel[1]:.3f}, {gt_lin_vel[2]:.3f}] | Est: [{est_lin_vel[0]:.3f}, {est_lin_vel[1]:.3f}, {est_lin_vel[2]:.3f}]")
            print(f"CoM-CoP | GT: [{gt_com_cop[0]:.3f}, {gt_com_cop[1]:.3f}, {gt_com_cop[2]:.3f}] | Est: [{est_com_cop[0]:.3f}, {est_com_cop[1]:.3f}, {est_com_cop[2]:.3f}]")
            print("-" * 20)


        obs[:,0:3] = estimated_targets[:,:3]
        obs[:,9:12] = estimated_targets[:,3:6]





        actions = student_policy(obs.detach())
        obs, _, rews, dones, infos, gt = env.step(actions.detach())

        elapsed = time.time() - start_time
        if elapsed < control_dt:
            time.sleep(control_dt - elapsed)
        

        

if __name__ == '__main__':
    EXPORT = False
    args = args_cli
    play(args)
