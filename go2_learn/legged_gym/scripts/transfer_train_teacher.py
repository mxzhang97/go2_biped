



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



parser = argparse.ArgumentParser(description="Train teacher/motion prior...")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--teacher_checkpoint", type=str, help = "Path to student checkpoint.")
parser.add_argument('--headless',       action='store_true', default=False)  
parser.add_argument('-c', '--cpu',      action='store_true', default=False) 
args_cli = parser.parse_args()



args_cli.device = "cpu" if args_cli.cpu else "cuda:0"


#params config
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
    entropy_coef_high: float = 0.01 
    entropy_coef_low: float = 0.01


    total_updates = total_timesteps // (num_envs * num_steps_per_rollout)

    entropy_anneal_start: int = int (total_updates * 0.40)
    entropy_anneal_end: int = int(total_updates * 0.60)


    #logging
    log_interval: int = 1000 #
    run_name: str = f"teacher_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir: str = f"logs/teacher_clean_smooth/{run_name}"
    save_interval: int = 5000 

cfg = TrainConfig()





class TeacherPolicy(nn.Module):
    def __init__(self, actor_obs_dim=52, privileged_obs_dim=2, action_dim=12):
        super().__init__()

        self.actor_network = nn.Sequential(nn.Linear(actor_obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128,action_dim))
        self.critic_network = nn.Sequential(nn.Linear(actor_obs_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128,1))
        self.actor_std = nn.Parameter(torch.ones(1, action_dim))

    
    def forward(self, actor_obs, privileged_obs):
        action_mean = self.actor_network(actor_obs)
        dist = Normal(action_mean, self.actor_std)
        value = self.critic_network(actor_obs)
        return dist, value, action_mean, self.actor_std

    def get_action_and_value(self, actor_obs, privileged_obs, action=None):
        dist, value, mu, sigma = self.forward(actor_obs, privileged_obs)
        if action is None: action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value, mu, sigma



class RolloutBuffer:
    def __init__(self, num_envs, num_steps, device, actor_obs_dim, critic_obs_dim, action_dim):
        self.device = device; self.num_steps = num_steps; self.num_envs = num_envs
        self.actor_obs_dim = actor_obs_dim; self.critic_obs_dim = critic_obs_dim; self.action_dim = action_dim
        self.reset()
    def reset(self):
        self.actor_obs = torch.zeros((self.num_steps, self.num_envs, self.actor_obs_dim), device=self.device)
        self.critic_obs = torch.zeros((self.num_steps, self.num_envs, self.critic_obs_dim), device=self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs, self.action_dim), device=self.device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.timeouts = torch.zeros((self.num_steps, self.num_envs), device=self.device)
        self.action_mus = torch.zeros((self.num_steps, self.num_envs, self.action_dim), device=self.device)
        self.action_sigmas = torch.zeros((self.num_steps, self.num_envs, self.action_dim), device=self.device)
        self.step = 0


    def add(self, actor_obs, critic_obs, action, log_prob, reward, done, value, timeout, mus, sigmas):
        self.actor_obs[self.step] = actor_obs
        self.critic_obs[self.step] = critic_obs
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

                #bootstrap
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






def load_teacher_policy(checkpoint_path, device):
    teacher_policy = TeacherPolicy().to(device)
    full_checkpoint = torch.load(checkpoint_path, map_location=device)
    teacher_policy.load_state_dict(full_checkpoint)
    print("Teacher policy loaded successfully.")
    return teacher_policy 



class KLAdaptiveLR:
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
        if kl_divergence > self.kl_threshold * self._kl_factor:
            self.lr = max(self.lr / self._lr_factor, self.min_lr)
        elif kl_divergence < self.kl_threshold / self._kl_factor:
            self.lr = min(self.lr * self._lr_factor, self.max_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr







def main():
    print("Starting training loop..")
    
    writer = SummaryWriter(cfg.log_dir)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)


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
    
    actor_obs_dim = env_cfg.env.num_observations; critic_obs_dim = env_cfg.env.num_observations; action_dim = env_cfg.env.num_actions
    teacher_policy = TeacherPolicy(actor_obs_dim, critic_obs_dim - actor_obs_dim, action_dim).to(cfg.device)

    optimizer = optim.Adam(teacher_policy.parameters(), lr=cfg.learning_rate)
    scheduler = KLAdaptiveLR(optimizer, cfg.learning_rate, cfg.kl_threshold, cfg.kl_factor, cfg.lr_factor, cfg.min_lr, cfg.max_lr)
    buffer = RolloutBuffer(cfg.num_envs, cfg.num_steps_per_rollout, cfg.device, actor_obs_dim, critic_obs_dim, action_dim)


    current_episode_rewards = torch.zeros(cfg.num_envs, device=cfg.device)
    current_episode_lengths = torch.zeros(cfg.num_envs, device=cfg.device)

    ep_rew_buffer = deque(maxlen=100)
    ep_len_buffer = deque(maxlen=100)

    rollout_reward_components = {}

    all_value_losses = []
    all_policy_losses = []
    all_entropy_losses = []
    all_total_losses = []

    all_lr = []
    all_kl_divs = []
    all_policy_update_alpha = []


    last_log_per_env_steps = 0
    last_save_step = 0

    if args_cli.teacher_checkpoint:
        print(f"Loading checkpoint from: {args_cli.teacher_checkpoint}")
        checkpoint = torch.load(args_cli.teacher_checkpoint, map_location=cfg.device)
        
        teacher_policy.load_state_dict(checkpoint['policy_state_dict'])

        print(f"Loading teacher successful: {args_cli.teacher_checkpoint}")
        if('optimizer_state_dict' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loading teacher optimizer successful: {args_cli.teacher_checkpoint}")
        
        # Load other training state variables
        # Set the scheduler's current LR to the saved value
        if('scheduler_lr' in checkpoint):
            scheduler.lr = checkpoint['scheduler_lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.lr
            print(f"Loading teacher LR successful: {args_cli.teacher_checkpoint}")
            

    global_step = 0
    start_time = time.time()
    obs, _ = env.reset()
    next_actor_obs, next_critic_obs = obs, obs.clone()
    next_done = torch.zeros(cfg.num_envs, device=cfg.device)

    print(f"\n--- Starting Training ---")
    print(f"Total Updates: {cfg.total_updates}")
    
    pbar = tqdm(range(1, cfg.total_updates + 1), desc="Training")
    for update in pbar:


        rl_coef=1.0
        

        if(update <= cfg.entropy_anneal_start): entropy_coef = cfg.entropy_coef_high
        elif (update <= cfg. entropy_anneal_end):progress = (update - cfg.entropy_anneal_start)/(cfg.entropy_anneal_end - cfg.entropy_anneal_start); entropy_coef = cfg.entropy_coef_high - (cfg.entropy_coef_high - cfg.entropy_coef_low) * progress
        else: entropy_coef = cfg.entropy_coef_low

        #rollouts
        buffer.reset()
        teacher_policy.eval()
        with torch.inference_mode():

            for step in range(cfg.num_steps_per_rollout):
                global_step += cfg.num_envs
                privileged_obs = next_critic_obs[:, actor_obs_dim:]
                
                action, log_prob, _, value, mu, sigma = teacher_policy.get_action_and_value(next_actor_obs, privileged_obs)



                
                next_obs, _, reward, done, info = env.step(action)



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

                
                buffer.add(next_actor_obs, next_critic_obs, action, log_prob, reward, done, value, info.get("time_outs", torch.zeros_like(done)),mu,sigma)
                next_actor_obs, next_critic_obs = next_obs, next_obs.clone()
        

            #GAE calc
            last_value = teacher_policy.get_action_and_value(next_actor_obs, next_critic_obs[:, actor_obs_dim:])[3]
            buffer.compute_returns_and_advantages(last_value.flatten(), cfg.gamma, cfg.gae_lambda)

        
        teacher_policy.train()
        data = buffer.get_data()
        advantages = data['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        kl_divs = []

        for epoch in range(cfg.num_epochs):
            idxs = torch.randperm(data['actions'].size(0))
            for start in range(0, data['actions'].size(0), cfg.batch_size):
                end = start + cfg.batch_size; batch_idxs = idxs[start:end]
                
                b_actor_obs = data['actor_obs'][batch_idxs]
                b_critic_obs = data['critic_obs'][batch_idxs]
                b_actions = data['actions'][batch_idxs]
                b_log_probs = data['log_probs'][batch_idxs]
                b_returns = data['returns'][batch_idxs]
                b_mus = data['action_mus'][batch_idxs]
                b_sigmas = data['action_sigmas'][batch_idxs]
                b_advantages = advantages[batch_idxs]
                
                dist, value, mu, sigma = teacher_policy(b_actor_obs, b_critic_obs[:, actor_obs_dim:])
                value = value.flatten()
                
                log_prob = dist.log_prob(b_actions).sum(-1)
                entropy = dist.entropy().sum(-1)


                b_values = data['values'][batch_idxs] 

                value_clipped = b_values + torch.clamp(
                    value - b_values,
                    -cfg.clip_eps,
                    cfg.clip_eps
                )
                
                value_loss_unclipped = (value - b_returns)**2
                value_loss_clipped = (value_clipped - b_returns)**2
                
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()



                
                ratio = torch.exp(log_prob - b_log_probs)
                pg_loss = -torch.min(ratio * b_advantages, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_advantages).mean()
                
                entropy_loss = -entropy.mean()
                total_loss = (rl_coef * (pg_loss + cfg.value_loss_coef * value_loss + entropy_coef * entropy_loss))



                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma / b_sigmas + 1.e-5) + (torch.square(b_sigmas) + torch.square(b_mus - mu)) / (2.0 * torch.square(sigma)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    kl_divs.append(kl_mean)

                    scheduler.step(kl_mean)


                
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(teacher_policy.parameters(), cfg.max_grad_norm)
                optimizer.step()


        #logging
        all_value_losses.append(value_loss.item())
        all_policy_losses.append(pg_loss.item())
        all_entropy_losses.append(-entropy_loss.item()) # Store positive entropy
        all_total_losses.append(total_loss.item())

        mean_kl_for_update = torch.stack(kl_divs).mean().item()
        all_kl_divs.append(mean_kl_for_update)

        all_lr.append(scheduler.lr)

        per_env_steps = int(global_step / cfg.num_envs)
        
        if per_env_steps >= last_log_per_env_steps + cfg.log_interval:

            mean_v_loss = np.mean(all_value_losses)
            mean_p_loss = np.mean(all_policy_losses)
            mean_e_loss = np.mean(all_entropy_losses)
            mean_t_loss = np.mean(all_total_losses)

            mean_kl_log = np.mean(all_kl_divs)
            mean_lr_log = np.mean(all_lr)

            with torch.inference_mode():

                mean_std = teacher_policy.actor_std.mean().item()
                writer.add_scalar("charts/policy_std_dev", mean_std, per_env_steps)


            writer.add_scalar("losses/total_loss", mean_t_loss, per_env_steps)
            writer.add_scalar("losses/value_loss", mean_v_loss, per_env_steps)
            writer.add_scalar("losses/policy_loss", mean_p_loss, per_env_steps)
            writer.add_scalar("losses/entropy_loss", mean_e_loss, per_env_steps)
            writer.add_scalar("losses/kl_div", mean_kl_log, per_env_steps)
            writer.add_scalar("losses/lr", mean_lr_log, per_env_steps)
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
            all_total_losses.clear()
            all_kl_divs.clear()
            all_lr.clear()
            all_policy_update_alpha.clear()

            rollout_reward_components.clear()
            
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
                'policy_state_dict': teacher_policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_lr': scheduler.lr 
            }
            torch.save(checkpoint, save_path)
            last_save_step = per_env_steps


    save_path = f"{cfg.log_dir}/checkpoints/agent_last.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'update': "last",
        'global_step': global_step,
        'policy_state_dict': teacher_policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_lr': scheduler.lr 
    }
    torch.save(checkpoint, save_path)

    writer.close()
    print("--- Training Finished ---")

if __name__ == "__main__":
    main()
    print("Exiting...")