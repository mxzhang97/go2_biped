

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry


import argparse
import os
import time
from collections import deque
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

import torch
torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser(description="Train a student policy via distillation.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--teacher_checkpoint", type=str, default="", help="Path to the teacher model checkpoint.")
parser.add_argument("--student_checkpoint", type=str, help = "Path to student checkpoint.")
parser.add_argument('--headless',       action='store_true', default=False)  
parser.add_argument('-c', '--cpu',      action='store_true', default=False)  
args_cli = parser.parse_args()



args_cli.device = "cpu" if args_cli.cpu else "cuda:0"


class TrainConfig:
    num_envs: int = args_cli.num_envs
    seed: int = args_cli.seed
    device: str = args_cli.device

    total_timesteps: int = 100000 * args_cli.num_envs
    num_steps_per_rollout: int = 24
    num_epochs: int = 5
    batch_size: int = (args_cli.num_envs * 24)//4
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    max_grad_norm: float = 1.0


    kl_threshold: float = 0.01
    kl_factor: float = 2.0  
    lr_factor: float = 1.5  
    min_lr: float = 1e-5
    max_lr: float = 1e-2


    value_loss_coef: float = 1.0
    imitation_loss_coef_low: float = 0.0 
    imitation_loss_coef_high: float = 5.0
    entropy_coef_high: float = 0.01 
    entropy_coef_low: float = 0.01


    total_updates = total_timesteps // (num_envs * num_steps_per_rollout)
    warmup_updates: int = int(total_updates * 0.30) 
    guided_updates: int = int(total_updates * 0.40)
    annealing_updates: int = int(total_updates * 0.60)
    entropy_anneal_start: int = int (total_updates * 0.40)
    entropy_anneal_end: int = int(total_updates * 0.60)


    log_interval: int = 1000 
    run_name: str = f"student_w_quad_quad_network_proprio_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir: str = f"logs/student_w_quad_clean_smooth/{run_name}"
    save_interval: int = 5000 

cfg = TrainConfig()


print("2. Defining network architectures...")
class TeacherPolicy(nn.Module):
    def __init__(self, actor_obs_dim=52, action_dim=12):
        super().__init__()
        self.actor_network = nn.Sequential(nn.Linear(actor_obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128,action_dim))
        self.critic_network = nn.Sequential(nn.Linear(actor_obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128,1))
        self.actor_std = nn.Parameter(torch.ones(1, action_dim))

    def forward(self, obs): 
        return self.actor_network(obs)

    @torch.no_grad()
    def get_action(self, obs):
        self.eval(); return self.forward(obs)

class StudentPolicy(nn.Module):
    def __init__(self, actor_obs_dim=52, privileged_obs_dim=2, action_dim=12):
        super().__init__()

        self.actor_network = nn.Sequential(nn.Linear(actor_obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128,action_dim))
        self.critic_network = nn.Sequential(nn.Linear(actor_obs_dim + privileged_obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128,1))


        self.actor_network_quad = nn.Sequential(nn.Linear(actor_obs_dim-6, 128), nn.ELU(), nn.Linear(128,128), nn.ELU(), nn.Linear(128,action_dim))
        self.critic_network_quad = nn.Sequential(nn.Linear(actor_obs_dim-6, 128), nn.ELU(), nn.Linear(128,128), nn.ELU(), nn.Linear(128,1))


        self.actor_std = nn.Parameter(torch.ones(1, action_dim))
        self.actor_quad_std = nn.Parameter(torch.ones(1, action_dim))

    
    def forward(self, actor_obs, privileged_obs):
        action_biped = self.actor_network(actor_obs)

        quad_mask = actor_obs[:,-1].unsqueeze(-1)

        action_mean =  quad_mask  * self.actor_network_quad(actor_obs[:,:-6]) + (1-quad_mask) * action_biped 

        std = quad_mask * self.actor_quad_std + (1-quad_mask) * self.actor_std

        dist =  Normal(action_mean, std) 

        value_biped = self.critic_network(torch.cat([actor_obs, privileged_obs], dim = -1))

        value = quad_mask * self.critic_network_quad(actor_obs[:,:-6]) + (1-quad_mask) * value_biped 


        return dist, value, action_mean, std 

    def get_action_and_value(self, actor_obs, privileged_obs, action=None):
        dist, value, mu, sigma = self.forward(actor_obs, privileged_obs)
        if action is None: action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value, mu, sigma


class StateEstimator(nn.Module):

    def __init__(self, proprio_dim=42*5, target_dim=6):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(proprio_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, target_dim))


    
    def forward(self, propio_history):

        pred = self.network(propio_history)

        return pred







class RolloutBuffer:
    def __init__(self, num_envs, num_steps, device, actor_obs_dim, critic_obs_dim, action_dim):
        self.device = device; self.num_steps = num_steps; self.num_envs = num_envs
        self.actor_obs_dim = actor_obs_dim; self.critic_obs_dim = critic_obs_dim; self.action_dim = action_dim
        self.reset()
    def reset(self):
        self.actor_obs = torch.zeros((self.num_steps, self.num_envs, self.actor_obs_dim), device=self.device)
        self.critic_obs = torch.zeros((self.num_steps, self.num_envs, self.critic_obs_dim), device=self.device)
        self.expert_actions = torch.zeros((self.num_steps, self.num_envs, self.action_dim), device=self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs, self.action_dim), device=self.device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.timeouts = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.action_mus = torch.zeros((self.num_steps, self.num_envs, self.action_dim), device=self.device)
        self.action_sigmas = torch.zeros((self.num_steps, self.num_envs, self.action_dim), device=self.device)
        self.step = 0


    def add(self, actor_obs, critic_obs, expert_action, action, log_prob, reward, done, value, timeout, mus, sigmas):
        self.actor_obs[self.step] = actor_obs
        self.critic_obs[self.step] = critic_obs
        self.expert_actions[self.step] = expert_action
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob.detach()
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value.flatten().detach()
        self.timeouts[self.step] = timeout
        self.action_mus[self.step] = mus
        self.action_sigmas[self.step] = sigmas
        self.step += 1


    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
            self.advantages = torch.zeros_like(self.rewards).to(self.device)
            last_gae_lam = 0
            
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_values = last_value
                else:
                    next_values = self.values[t + 1]

                #bootstrap with current critic network value estimate for timeouts
                next_values = torch.where(self.timeouts[t].bool(), self.values[t], next_values)
                
                next_is_failure = self.dones[t] * (1 - self.timeouts[t])
                next_non_terminal = 1.0 - next_is_failure

                delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
                self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
            self.returns = self.advantages + self.values


    def get_data(self):
        num_transitions = self.num_steps * self.num_envs
        all_tensors = {k: v for k, v in self.__dict__.items() if isinstance(v, torch.Tensor)}
        return {k: v.reshape(num_transitions, -1) if v.dim() > 2 else v.reshape(-1) for k, v in all_tensors.items()}




class RolloutStateEstimator:
    def __init__(self, num_envs, num_steps, device, proprio_dim, target_dim, history_len):
        self.device = device
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.proprio_dim = proprio_dim
        self.target_dim = target_dim
        self.history_len = history_len

        self.reset()

    def reset(self):

        self.proprios = torch.zeros((self.num_steps, self.num_envs, self.proprio_dim), device=self.device)
        self.targets = torch.zeros((self.num_steps, self.num_envs, self.target_dim), device=self.device)
        self.history_frame = torch.zeros((self.num_steps - self.history_len, self.num_envs,self.proprio_dim * self.history_len), device=self.device)

        self.step = 0

        if hasattr(self, 'history_frames'):
            del self.history_frames
        if hasattr(self, 'history_targets'):
            del self.history_targets


    def add(self, proprio, target):

        self.proprios[self.step] = proprio 
        self.targets[self.step] =target 
        self.step += 1

    
    def process_frames(self):

        # This is a highly efficient, vectorized way to create sliding windows.
        # We use `unfold` to create windows of size `history_len` along the time dimension (dim=0).
        # Shape: (num_steps, num_envs, proprio_dim) -> (num_windows, num_envs, proprio_dim, history_len)
        unfolded_proprios = self.proprios.unfold(dimension=0, size=self.history_len, step=1)

        # We need to reshape this into the final desired format: (num_windows, num_envs, history_len * proprio_dim)
        # First, permute to bring the history_len dimension next to the proprio_dim for flattening.
        # Shape: (num_windows, num_envs, proprio_dim, history_len) -> (num_windows, num_envs, history_len, proprio_dim)
        permuted_proprios = unfolded_proprios.permute(0, 1, 3, 2)

        # Now, flatten the last two dimensions to create the feature vector for the MLP.
        # The number of usable windows is num_steps - history_len + 1.
        num_windows = permuted_proprios.shape[0]
        self.history_frames = permuted_proprios.reshape(num_windows, self.num_envs, -1)

        # The target for each window corresponds to the LAST frame in that window.
        # The first window uses frames [0, 1, ..., history_len-1], so its target is at index `history_len-1`.
        # We can get all targets by slicing the original targets tensor.
        self.history_targets = self.targets[self.history_len - 1:, :, :]




    def get_data(self):

        num_samples = self.history_frames.shape[0] * self.history_frames.shape[1]
        
        # Reshape the tensors to have a single batch dimension.
        # Shape: (num_windows, num_envs, features) -> (num_samples, features)
        batch_history_frames = self.history_frames.reshape(num_samples, -1)
        batch_history_targets = self.history_targets.reshape(num_samples, -1)

        # Create a random permutation to shuffle the data before training.
        # This is crucial to break temporal correlations between batches.
        shuffled_indices = torch.randperm(num_samples, device=self.device)

        return {
            'history_frames': batch_history_frames[shuffled_indices],
            'targets': batch_history_targets[shuffled_indices]
        }




def load_teacher_policy(checkpoint_path, device, obs_dim, action_dim):
    teacher_policy = TeacherPolicy(obs_dim, action_dim).to(device)
    full_checkpoint = torch.load(checkpoint_path, map_location=device)
    teacher_policy.load_state_dict(full_checkpoint['policy_state_dict'])
    print("Teacher policy loaded successfully.")
    return teacher_policy




class KLAdaptiveLR:
    """A scheduler that adapts the LR based on a target KL divergence, matching skrl's logic."""
    def __init__(self, optimizer, initial_lr, kl_threshold, kl_factor, lr_factor, min_lr, max_lr):
        self.optimizer = optimizer
        self.kl_threshold = kl_threshold
        self._kl_factor = kl_factor
        self._lr_factor = lr_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        self.lr = initial_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def step(self, kl_divergence):
        """Adjusts the learning rate for all optimizer parameter groups."""
        if kl_divergence > self.kl_threshold * self._kl_factor:
            self.lr = max(self.lr / self._lr_factor, self.min_lr)
        elif kl_divergence < self.kl_threshold / self._kl_factor:
            self.lr = min(self.lr * self._lr_factor, self.max_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr






def main():
    print("Starting training loop..")
    
    writer = SummaryWriter(cfg.log_dir)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)


    env_cfg = task_registry.get_cfgs(name=args_cli.task, seed=cfg.seed)

    if(args_cli.num_envs):
        env_cfg.env.num_envs = args_cli.num_envs


    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    if env_cfg.terrain.mesh_type == "plane":
        for i in range(2):
            env_cfg.viewer.pos[i] = env_cfg.viewer.pos[i] - env_cfg.terrain.plane_length / 4
            env_cfg.viewer.lookat[i] = env_cfg.viewer.lookat[i] - env_cfg.terrain.plane_length / 4

    env = task_registry.make_env(name=args_cli.task, args=args_cli, env_cfg=env_cfg)
    obs = env.get_observations()
    
    env_cfg.seed = cfg.seed
    
    actor_obs_dim = env_cfg.env.num_observations; critic_obs_dim = env_cfg.env.num_privileged_observations; action_dim = env_cfg.env.num_actions

    student_policy = StudentPolicy(actor_obs_dim, critic_obs_dim, action_dim).to(cfg.device)

    teacher_policy = load_teacher_policy(args_cli.teacher_checkpoint, cfg.device, actor_obs_dim + critic_obs_dim -1, action_dim)


    state_estimator = StateEstimator(42*5, 6).to(cfg.device)


    biped_params = []
    biped_params.extend(student_policy.actor_network.parameters())
    biped_params.extend(student_policy.critic_network.parameters())
    biped_params.append(student_policy.actor_std)


    quad_params = []
    quad_params.extend(student_policy.actor_network_quad.parameters())
    quad_params.extend(student_policy.critic_network_quad.parameters())
    quad_params.append(student_policy.actor_quad_std)





    optimizer = optim.Adam(biped_params, lr=cfg.learning_rate)
    optimizer_quad = optim.Adam(quad_params, lr = cfg.learning_rate)
    optimizer_state_estimator = optim.Adam(state_estimator.network.parameters(), lr = 1e-4)



    scheduler = KLAdaptiveLR(optimizer, cfg.learning_rate, cfg.kl_threshold, cfg.kl_factor, cfg.lr_factor, cfg.min_lr, cfg.max_lr)
    scheduler_quad = KLAdaptiveLR(optimizer_quad, cfg.learning_rate, cfg.kl_threshold, cfg.kl_factor, cfg.lr_factor, cfg.min_lr, cfg.max_lr)


    buffer = RolloutBuffer(cfg.num_envs, cfg.num_steps_per_rollout, cfg.device, actor_obs_dim, critic_obs_dim, action_dim)
    se_buffer = RolloutStateEstimator(cfg.num_envs,cfg.num_steps_per_rollout,cfg.device,42,6,5)



    current_episode_rewards = torch.zeros(cfg.num_envs, device=cfg.device)
    current_episode_lengths = torch.zeros(cfg.num_envs, device=cfg.device)

    ep_rew_buffer = deque(maxlen=100)
    ep_len_buffer = deque(maxlen=100)

    rollout_reward_components = {}

    all_value_losses = []
    all_policy_losses = []
    all_entropy_losses = []
    all_imitation_losses = []
    all_total_losses = []

    all_lr = []
    all_kl_divs = []
    all_policy_update_alpha = []



    all_value_losses_quad = []
    all_policy_losses_quad = []
    all_entropy_losses_quad = []
    all_imitation_losses_quad = []
    all_total_losses_quad = []

    all_lr_quad = []
    all_kl_divs_quad = []
    all_policy_update_alpha_quad = []

    se_mse = []



    last_log_per_env_steps = 0
    last_save_step = 0



    if args_cli.student_checkpoint:
        print(f"Loading checkpoint from: {args_cli.student_checkpoint}")
        checkpoint = torch.load(args_cli.student_checkpoint, map_location=cfg.device)
        
        student_policy.load_state_dict(checkpoint['policy_state_dict'])

        print(f"Loading student successful: {args_cli.student_checkpoint}")
        if('optimizer_state_dict' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loading student optimizer successful: {args_cli.student_checkpoint}")

        if('optimizer_quad_state_dict' in checkpoint):
            optimizer_quad.load_state_dict(checkpoint['optimizer_quad_state_dict'])
            print(f"Loading student optimizer successful: {args_cli.student_checkpoint}")

        if('scheduler_lr' in checkpoint):
            scheduler.lr = checkpoint['scheduler_lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.lr
            print(f"Loading student LR successful: {args_cli.student_checkpoint}")

        if('scheduler_quad_lr' in checkpoint):
            scheduler_quad.lr = checkpoint['scheduler_quad_lr']
            for param_group in optimizer_quad.param_groups:
                param_group['lr'] = scheduler.lr
            print(f"Loading student LR successful: {args_cli.student_checkpoint}")
            



    global_step = 0
    start_time = time.time()
    obs, privileged_obs,gt  = env.reset()
    next_actor_obs, next_critic_obs = obs, privileged_obs

    print(f"\n--- Starting Training ---")
    print(f"Total Updates: {cfg.total_updates}, Warm-up until: {cfg.warmup_updates}, Guided until: {cfg.guided_updates}")
    
    pbar = tqdm(range(1, cfg.total_updates + 1), desc="Training")
    for update in pbar:


        
        if update <= cfg.warmup_updates: rl_coef = 1.0; imitation_coef=0.0
        elif update <= cfg.guided_updates: rl_coef = 1.0; progress = (update - cfg.warmup_updates) / (cfg.guided_updates - cfg.warmup_updates); imitation_coef =cfg.imitation_loss_coef_low + (cfg.imitation_loss_coef_high - cfg.imitation_loss_coef_low) * progress
        else: rl_coef = 1.0; imitation_coef = cfg.imitation_loss_coef_high

        
        if(update <= cfg.entropy_anneal_start): entropy_coef = cfg.entropy_coef_high
        elif (update <= cfg. entropy_anneal_end):progress = (update - cfg.entropy_anneal_start)/(cfg.entropy_anneal_end - cfg.entropy_anneal_start); entropy_coef = cfg.entropy_coef_high - (cfg.entropy_coef_high - cfg.entropy_coef_low) * progress
        else: entropy_coef = cfg.entropy_coef_low


        #rollout
        buffer.reset()
        se_buffer.reset()
        student_policy.eval()
        with torch.inference_mode():

            for step in range(cfg.num_steps_per_rollout):
                global_step += cfg.num_envs
                
                action, log_prob, _, value, mu, sigma = student_policy.get_action_and_value(next_actor_obs, next_critic_obs)
                expert_action = teacher_policy.get_action(torch.cat([next_actor_obs[:,:-1], next_critic_obs], dim=-1))




                if(update >=cfg.annealing_updates):

                    #casing on footheight
                    rear_foot_indices = env.simulator.feet_indices[2:]

                    foot_heights = env.simulator.rigid_body_states.view(env.num_envs, env.simulator.num_bodies, 13)[:, rear_foot_indices, 2]

                    foot_heights_mask = (torch.any(foot_heights>0.04, dim=-1)) #just using 0.04 as a test, 0.03 gets the full gait, still probably want 0.03 with the zero vel cases in.
                    expert_action = torch.where(foot_heights_mask.unsqueeze(-1), expert_action, action)

                
                expert_action = torch.where((torch.norm(next_actor_obs[:,-6:-3], dim=-1)<5e-1).unsqueeze(-1), action, expert_action)


                expert_action = torch.where(next_actor_obs[:,-1].bool().unsqueeze(-1), action, expert_action)





                
                next_obs, next_privileged_obs, reward, done, info, gt = env.step(action)






                if 'episode' in info:
                    for key, val in info['episode'].items():
                        rollout_reward_components.setdefault(key, []).append(val)


                current_episode_rewards += reward
                current_episode_lengths += 1
                
                if torch.any(done):
                    finished_env_indices = done.nonzero(as_tuple=False).flatten()
                    
                    ep_rew_buffer.extend(current_episode_rewards[finished_env_indices].tolist())
                    ep_len_buffer.extend(current_episode_lengths[finished_env_indices].tolist())
                    
                    current_episode_rewards[finished_env_indices] = 0
                    current_episode_lengths[finished_env_indices] = 0


                time_outs = info.get("time_outs", torch.zeros_like(done))

                mode_switches = (next_actor_obs[:, -1] != next_obs[:, -1])

                effective_timeouts = time_outs | mode_switches


                
                buffer.add(next_actor_obs, next_critic_obs, expert_action, action, log_prob, reward, done, value, effective_timeouts ,mu,sigma)


                imu = next_actor_obs[:,3:9]
                joints_last_action = next_actor_obs[:,12:48]
                proprio = torch.concat([imu,joints_last_action], dim=-1)

                se_buffer.add(proprio, gt)

                next_actor_obs, next_critic_obs = next_obs, next_privileged_obs
        

            #GAe
            last_value = student_policy.get_action_and_value(next_actor_obs, next_privileged_obs)[3]
            buffer.compute_returns_and_advantages(last_value.flatten(), cfg.gamma, cfg.gae_lambda)

            se_buffer.process_frames()


        #update
        
        student_policy.train()
        data = buffer.get_data()

        #advantages = data['advantages']
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        kl_divs = []
        kl_divs_quad = []

        for epoch in range(cfg.num_epochs):
            idxs = torch.randperm(data['actions'].size(0))
            for start in range(0, data['actions'].size(0), cfg.batch_size):
                end = start + cfg.batch_size; batch_idxs = idxs[start:end]
                
                b_actor_obs = data['actor_obs'][batch_idxs]
                b_critic_obs = data['critic_obs'][batch_idxs]
                b_expert_actions = data['expert_actions'][batch_idxs]
                b_actions = data['actions'][batch_idxs]
                b_log_probs = data['log_probs'][batch_idxs]
                b_returns = data['returns'][batch_idxs]
                b_mus = data['action_mus'][batch_idxs]
                b_sigmas = data['action_sigmas'][batch_idxs]
                b_advantages = data['advantages'][batch_idxs]
                b_values = data['values'][batch_idxs]
                

                #seperate calcs for the different modes 
                quad_mode_mask = b_actor_obs[:,-1].bool()
                biped_mode_mask = ~quad_mode_mask

                b_actor_obs_biped = b_actor_obs[biped_mode_mask]
                b_critic_obs_biped = b_critic_obs[biped_mode_mask]
                b_expert_actions_biped = b_expert_actions[biped_mode_mask]
                b_actions_biped = b_actions[biped_mode_mask]
                b_log_probs_biped = b_log_probs[biped_mode_mask]
                b_returns_biped = b_returns[biped_mode_mask]
                b_mus_biped = b_mus[biped_mode_mask]
                b_sigmas_biped = b_sigmas[biped_mode_mask]
                b_advantages_biped = b_advantages[biped_mode_mask]
                b_values_biped = b_values[biped_mode_mask]



                b_actor_obs_quad = b_actor_obs[quad_mode_mask]
                b_critic_obs_quad = b_critic_obs[quad_mode_mask]
                b_expert_actions_quad = b_expert_actions[quad_mode_mask]
                b_actions_quad = b_actions[quad_mode_mask]
                b_log_probs_quad = b_log_probs[quad_mode_mask]
                b_returns_quad = b_returns[quad_mode_mask]
                b_mus_quad = b_mus[quad_mode_mask]
                b_sigmas_quad = b_sigmas[quad_mode_mask]
                b_advantages_quad = b_advantages[quad_mode_mask]
                b_values_quad = b_values[quad_mode_mask]


                if biped_mode_mask.any():

                    
                    b_advantages_biped = (b_advantages_biped - b_advantages_biped.mean())/(b_advantages_biped.std() + 1e-8)

                    dist_biped, value_biped, mu_biped, sigma_biped = student_policy(b_actor_obs_biped, b_critic_obs_biped)
                    value_biped = value_biped.flatten()

                    log_prob_biped = dist_biped.log_prob(b_actions_biped).sum(-1)
                    entropy_biped = dist_biped.entropy().sum(-1)


                    value_clipped_biped = b_values_biped + torch.clamp(
                        value_biped - b_values_biped,
                        -cfg.clip_eps,
                        cfg.clip_eps
                    )



                    value_loss_unclipped_biped = (value_biped - b_returns_biped)**2
                    value_loss_clipped_biped = (value_clipped_biped - b_returns_biped)**2


                    value_loss_biped = torch.max(value_loss_unclipped_biped, value_loss_clipped_biped).mean()

                    imitation_loss_biped = ((dist_biped.mean - b_expert_actions_biped) ** 2).mean()

                
                    ratio_biped = torch.exp(log_prob_biped - b_log_probs_biped)
                    pg_loss_biped = -torch.min(ratio_biped * b_advantages_biped, torch.clamp(ratio_biped, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_advantages_biped).mean()

                    entropy_loss_biped = -entropy_biped.mean()


                    total_loss_biped = (rl_coef * (pg_loss_biped + cfg.value_loss_coef * value_loss_biped + entropy_coef * entropy_loss_biped) + imitation_coef * imitation_loss_biped)

                    with torch.inference_mode():

                        #biped network
                        kl_biped = torch.sum(
                            torch.log(sigma_biped / b_sigmas_biped + 1.e-5) + (torch.square(b_sigmas_biped) + torch.square(b_mus_biped - mu_biped)) / (2.0 * torch.square(sigma_biped)) - 0.5, axis=-1)
                        kl_mean_biped = torch.mean(kl_biped)
                        kl_divs.append(kl_mean_biped)

                        scheduler.step(kl_mean_biped)


                    optimizer.zero_grad()
                    total_loss_biped.backward()
                    nn.utils.clip_grad_norm_(biped_params, cfg.max_grad_norm)
                    optimizer.step()

                if quad_mode_mask.any():



                    b_advantages_quad = (b_advantages_quad - b_advantages_quad.mean())/(b_advantages_quad.std() + 1e-8)


                    
                    dist_quad, value_quad, mu_quad, sigma_quad = student_policy(b_actor_obs_quad, b_critic_obs_quad)
                    value_quad = value_quad.flatten()




                    log_prob_quad = dist_quad.log_prob(b_actions_quad).sum(-1)
                    entropy_quad = dist_quad.entropy().sum(-1)



                    value_clipped_quad = b_values_quad + torch.clamp(
                        value_quad - b_values_quad,
                        -cfg.clip_eps,
                        cfg.clip_eps
                    )
                    
                    value_loss_unclipped_quad = (value_quad - b_returns_quad)**2
                    value_loss_clipped_quad = (value_clipped_quad - b_returns_quad)**2



                    

                    value_loss_quad = torch.max(value_loss_unclipped_quad, value_loss_clipped_quad).mean()




                    #this should be 0 but just for consistency
                    imitation_loss_quad = ((dist_quad.mean - b_expert_actions_quad) ** 2).mean()





                    ratio_quad = torch.exp(log_prob_quad - b_log_probs_quad)
                    pg_loss_quad = -torch.min(ratio_quad * b_advantages_quad, torch.clamp(ratio_quad, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_advantages_quad).mean()



                    
                    entropy_loss_quad = -entropy_quad.mean()


                    total_loss_quad = (rl_coef * (pg_loss_quad + cfg.value_loss_coef * value_loss_quad + entropy_coef * entropy_loss_quad)) #+ imitation_coef * imitation_loss_quad)



                    with torch.inference_mode():

                        kl_quad = torch.sum(
                            torch.log(sigma_quad / b_sigmas_quad + 1.e-5) + (torch.square(b_sigmas_quad) + torch.square(b_mus_quad - mu_quad)) / (2.0 * torch.square(sigma_quad)) - 0.5, axis=-1)
                        kl_mean_quad = torch.mean(kl_quad)
                        kl_divs_quad.append(kl_mean_quad)

                        scheduler_quad.step(kl_mean_quad)


                

                    optimizer_quad.zero_grad()
                    total_loss_quad.backward()
                    nn.utils.clip_grad_norm_(quad_params, cfg.max_grad_norm)


                    optimizer_quad.step()



        ## SE train
        num_se_epochs = 4
        se_batch_size = 1024

        se_data = se_buffer.get_data()
        num_samples = se_data['history_frames'].shape[0]


        for epoch in range(num_se_epochs):
            for start in range(0, num_samples, se_batch_size):
                end = start + se_batch_size
                
                batch_history_frames = se_data['history_frames'][start:end]
                batch_targets = se_data['targets'][start:end]


                se_preds = state_estimator(batch_history_frames)
                se_loss = F.mse_loss(se_preds, batch_targets)

                optimizer_state_estimator.zero_grad()
                se_loss.backward()
                optimizer_state_estimator.step()

                se_mse.append(se_loss.item())





        #logging
        if(value_loss_biped.any()):

            all_value_losses.append(value_loss_biped.item())
            all_policy_losses.append(pg_loss_biped.item())
            all_entropy_losses.append(-entropy_loss_biped.item()) 
            all_total_losses.append(total_loss_biped.item())
            all_imitation_losses.append(imitation_loss_biped.item())
            

            mean_kl_for_update = torch.stack(kl_divs).mean().item()
            all_kl_divs.append(mean_kl_for_update)

            all_lr.append(scheduler.lr)
            all_policy_update_alpha.append(imitation_coef)
        
        if(value_loss_quad.any()):



            all_value_losses_quad.append(value_loss_quad.item())
            all_policy_losses_quad.append(pg_loss_quad.item())
            all_entropy_losses_quad.append(-entropy_loss_quad.item()) 
            all_total_losses_quad.append(total_loss_quad.item())
            all_imitation_losses_quad.append(imitation_loss_quad.item())
        

            mean_kl_for_update_quad = torch.stack(kl_divs_quad).mean().item()
            all_kl_divs_quad.append(mean_kl_for_update_quad)

            all_lr_quad.append(scheduler_quad.lr)
            all_policy_update_alpha_quad.append(imitation_coef) #redundant but yeah... it's late.


        per_env_steps = int(global_step / cfg.num_envs)
        
        if per_env_steps >= last_log_per_env_steps + cfg.log_interval:

            if(all_value_losses):

                mean_v_loss = np.mean(all_value_losses)
                mean_p_loss = np.mean(all_policy_losses)
                mean_e_loss = np.mean(all_entropy_losses)
                mean_i_loss = np.mean(all_imitation_losses)
                mean_t_loss = np.mean(all_total_losses)

                mean_kl_log = np.mean(all_kl_divs)
                mean_lr_log = np.mean(all_lr)
                mean_alpha_log = np.mean(all_policy_update_alpha)


            if(all_value_losses_quad):


                mean_v_loss_quad = np.mean(all_value_losses_quad)
                mean_p_loss_quad = np.mean(all_policy_losses_quad)
                mean_e_loss_quad = np.mean(all_entropy_losses_quad)
                mean_i_loss_quad = np.mean(all_imitation_losses_quad)
                mean_t_loss_quad = np.mean(all_total_losses_quad)

                mean_kl_log_quad = np.mean(all_kl_divs_quad)
                mean_lr_log_quad = np.mean(all_lr_quad)
                mean_alpha_log_quad = np.mean(all_policy_update_alpha_quad) #redundant 

            with torch.inference_mode():

                mean_std = student_policy.actor_std.mean().item()
                writer.add_scalar("charts/policy_std_dev", mean_std, per_env_steps)
                mean_std_quad = student_policy.actor_quad_std.mean().item()
                writer.add_scalar("charts/policy_quad_std_dev", mean_std_quad, per_env_steps)


            writer.add_scalar("losses/total_loss", mean_t_loss, per_env_steps)
            writer.add_scalar("losses/value_loss", mean_v_loss, per_env_steps)
            writer.add_scalar("losses/policy_loss", mean_p_loss, per_env_steps)
            writer.add_scalar("losses/imitation_loss", mean_i_loss, per_env_steps)
            writer.add_scalar("losses/entropy_loss", mean_e_loss, per_env_steps)
            writer.add_scalar("losses/kl_div", mean_kl_log, per_env_steps)
            writer.add_scalar("losses/lr", mean_lr_log, per_env_steps)
            writer.add_scalar("losses/imitation_coef", mean_alpha_log, per_env_steps)


            writer.add_scalar("losses/total_loss_quad", mean_t_loss_quad, per_env_steps)
            writer.add_scalar("losses/value_loss_quad", mean_v_loss_quad, per_env_steps)
            writer.add_scalar("losses/policy_loss_quad", mean_p_loss_quad, per_env_steps)
            writer.add_scalar("losses/imitation_loss_quad", mean_i_loss_quad, per_env_steps)
            writer.add_scalar("losses/entropy_loss_quad", mean_e_loss_quad, per_env_steps)
            writer.add_scalar("losses/kl_div_quad", mean_kl_log_quad, per_env_steps)
            writer.add_scalar("losses/lr_quad", mean_lr_log_quad, per_env_steps)
            writer.add_scalar("losses/imitation_coef_quad", mean_alpha_log_quad, per_env_steps)

            mean_se_loss = np.mean(se_mse)

            writer.add_scalar("losses/se_loss", mean_se_loss, per_env_steps)


            with torch.inference_mode():
                for key, values in rollout_reward_components.items():
                    if values:
                        stacked_values = torch.stack(values)
                        mean_value = stacked_values.mean().item()
                        writer.add_scalar("episode/" + key, mean_value, per_env_steps)


            if len(ep_rew_buffer) > 0:
                writer.add_scalar("charts/episodic_return_mean", np.mean(ep_rew_buffer), per_env_steps)
                writer.add_scalar("charts/episodic_length_mean", np.mean(ep_len_buffer), per_env_steps)

            else:
                writer.add_scalar("charts/episodic_return_mean", 0.0, per_env_steps)
                writer.add_scalar("charts/episodic_length_mean", 0.0, per_env_steps)

            all_value_losses.clear()
            all_policy_losses.clear()
            all_entropy_losses.clear()
            all_imitation_losses.clear()
            all_total_losses.clear()
            all_kl_divs.clear()
            all_lr.clear()
            all_policy_update_alpha.clear()


            all_value_losses_quad.clear()
            all_policy_losses_quad.clear()
            all_entropy_losses_quad.clear()
            all_imitation_losses_quad.clear()
            all_total_losses_quad.clear()
            all_kl_divs_quad.clear()
            all_lr_quad.clear()
            all_policy_update_alpha_quad.clear()

            rollout_reward_components.clear()

            se_mse.clear()
            
            last_log_per_env_steps = per_env_steps


        sps = int(global_step / (time.time() - start_time))
        it_s = int(per_env_steps/(time.time() - start_time))
        pbar.set_postfix({
            "SPS": f"{sps}",
            "it/s": f"{it_s}",
            "per_env_steps": f"{per_env_steps}"
        })


        if per_env_steps > last_save_step + cfg.save_interval:
            save_path = f"{cfg.log_dir}/checkpoints/agent_{per_env_steps}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint = {
                'update': update,
                'global_step': global_step,
                'policy_state_dict': student_policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_quad_state_dict': optimizer_quad.state_dict(),
                'scheduler_lr': scheduler.lr, 
                'scheduler_quad_lr': scheduler_quad.lr 
            }
            torch.save(checkpoint, save_path)
            last_save_step = per_env_steps


    save_path = f"{cfg.log_dir}/checkpoints/agent_last.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'update': "last",
        'global_step': global_step,
        'policy_state_dict': student_policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_quad_state_dict': optimizer_quad.state_dict(),
        'scheduler_lr': scheduler.lr,
        'scheduler_quad_lr': scheduler_quad.lr
    }
    torch.save(checkpoint, save_path)


    save_path = f"{cfg.log_dir}/se/last.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'update': "last",
        'global_step': global_step,
        'policy': state_estimator.state_dict(),
    }
    torch.save(checkpoint, save_path)

    writer.close()
    print("--- Training Finished ---")

if __name__ == "__main__":
    main()
    print("Exiting...")