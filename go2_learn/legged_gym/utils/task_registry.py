import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import random

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .utils import class_to_dict


from abc import ABC, abstractmethod
import torch
from typing import Tuple, Union

# minimal interface of the environment
class VecEnv(ABC):
    num_envs: int
    num_obs: int
    num_privileged_obs: int
    num_actions: int
    max_episode_length: int
    privileged_obs_buf: torch.Tensor
    obs_buf: torch.Tensor 
    rew_buf: torch.Tensor
    reset_buf: torch.Tensor
    episode_length_buf: torch.Tensor # current episode duration
    extras: dict
    device: torch.device
    @abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        pass
    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        pass
    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        pass
    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        pass

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name, seed):
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = seed
        return env_cfg


    def make_env(self, name, args=None, env_cfg=None):


        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)

        self._set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_device = "cpu" if args.cpu else "cuda:0"
        # sim_params
        sim_params = class_to_dict(env_cfg.sim)

        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            sim_device=sim_device,
                            headless=args.headless)
        return env

    def _set_seed(self,seed):
        if seed == -1:
            seed = np.random.randint(0, 10000)
        print("Setting seed: {}".format(seed))
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# make global task registry
task_registry = TaskRegistry()