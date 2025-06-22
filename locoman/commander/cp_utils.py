import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)

from typing import Dict, List, Tuple, Callable
import torch
import torch.nn as nn
import dill
import hydra
from omegaconf import OmegaConf
from consistency_policy.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
import re
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as T
# import pytorch3d.transforms as pt
import numpy as np
from diffusion_policy.common.pytorch_util import dict_apply
from collections import deque

NORMALIZER_PREFIX_LENGTH = 11
MODEL_PREFIX_LENGTH = 6

"""Next 2 Utils from the original CM implementation"""

@torch.no_grad()
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

@torch.no_grad()
def reduce_dims(x, target_dims):
    """Reduces dimensions from the end of a tensor until it has target_dims dimensions."""
    dims_to_reduce = x.ndim - target_dims
    if dims_to_reduce < 0:
         raise ValueError(
             f"input has {x.ndim} dims but target_dims is {target_dims}, which is greater"
         )
    for _ in range(dims_to_reduce):
        x = x.squeeze(-1)
    
    return x

def euler_to_quat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_quat()

# def rot6d_to_rmat(rot_6d: torch.Tensor) -> torch.Tensor:
#     return pt.rotation_6d_to_matrix(rot_6d)

def rmat_to_euler(rot_mat: np.ndarray, degrees=False) -> np.ndarray:
    if isinstance(rot_mat, torch.Tensor):
        rot_mat = rot_mat.cpu().numpy()
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler

def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(rot_mat).as_quat()
    return quat

def state_dict_to_model(state_dict, pattern=r'model\.'):
    new_state_dict = {}
    prefix = re.compile(pattern)

    for k, v in state_dict["state_dicts"]["model"].items():
        if re.match(prefix, k):
            # Remove prefix
            new_k = k[MODEL_PREFIX_LENGTH:]  
            new_state_dict[new_k] = v

    return new_state_dict

def load_normalizer(workspace_state_dict):
    keys = workspace_state_dict['state_dicts']['model'].keys()
    normalizer_keys = [key for key in keys if 'normalizer' in key]
    normalizer_dict = {key[NORMALIZER_PREFIX_LENGTH:]: workspace_state_dict['state_dicts']['model'][key] for key in normalizer_keys}

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(normalizer_dict)

    return normalizer

def get_policy(ckpt_path, cfg = None, dataset_path = None):
    """
    Returns loaded policy from checkpoint
    If cfg is None, the ckpt's saved cfg will be used
    """
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg'] if cfg is None else cfg

    cfg.training.inference_mode = True
    cfg.training.online_rollouts = False

    if dataset_path is not None:
        cfg.task.dataset.dataset_path = dataset_path
        cfg.task.envrunner.dataset_path = dataset_path

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_checkpoint(path=ckpt_path, exclude_keys=['optimizer'])
    workspace_state_dict = torch.load(ckpt_path)
    normalizer = load_normalizer(workspace_state_dict)

    policy = workspace.model
    policy.set_normalizer(normalizer)

    return policy

def get_cfg(ckpt_path):
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    return cfg

class IsaacPolicyWrapper:
    def __init__(self, policy, n_obs, n_acts, d_pos=6, d_rot=6, cfg=None, device="cpu"):
        self.policy = policy

        self.obs_chunker = ObsChunker(n_obs_steps=n_obs)
        self.obs_chunker.reset()

        self.action_chunker = ActionChunker(n_act_steps=n_acts)
        self.action_chunker.reset()

        self.cfg = cfg
        self.device = device

        self.c_obs = 0
        self.c_acts = 0

        self.d_pos = d_pos
        self.d_rot = d_rot

        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.to(self.device)),
            # add an unsqueeze to make it a batch of 1
            T.Lambda(lambda x: x.unsqueeze(0)) #.unsqueeze(0))
        ])


    def get_action(self, observation):
        action = self.action_chunker.get_action()
        if action is None:
            #we need to calculate actions

            # TODO: load this from shape_meta cfg rather than hardcoding
            shape_meta = self.cfg.shape_meta
            obs_shape_meta = shape_meta['obs']
            obs_dict = {
                key: observation[key] for key in obs_shape_meta.keys()
            }
            # obs_dict = {
            #     "base_pose": observation["base_pose"],
            #     "arm_pos": observation["arm_pos"],
            #     "arm_quat": observation["arm_quat"],
            #     "gripper_pos": observation["gripper_pos"],
            #     "base_image": observation["base_image"],
            #     "wrist_image": observation["wrist_image"],
            # }

            # transform image data
            for key in obs_dict:
                if "image" in key or "rgb" in key or "depth" in key:
                    obs_dict[key] = self.transform(obs_dict[key])
                else:
                    # add an unsqueeze to make it a batch of 1
                    obs_dict[key] = torch.tensor(obs_dict[key]).to(self.device).unsqueeze(0) #.unsqueeze(0)


            # assert not torch.any(obs_dict['arm_quat'][0] < 0), 'quaternion with negative value on first entry found' \
            #                                                 'policy learning assumes non-negative quat representation'

            # convert all values in obs_dict to numpy
            obs_dict = dict_apply(obs_dict, lambda x: x.cpu().numpy())
            self.obs_chunker.add_obs(obs_dict)
            obs = self.obs_chunker.get_obs_history()


            obs_dict_torch = dict_apply(obs,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
            )

            result = self.policy.predict_action(obs_dict_torch)

            # Hardcoded action shapes
            actions = result["action"]
            # pos = actions[..., :self.d_pos].cpu().numpy()
            # gripper = actions[..., [-1]].cpu().numpy()

            # rot = actions[..., self.d_pos: self.d_pos + self.d_rot]
            # # rot = rot6d_to_rmat(rot)
            # rot = rot[0].cpu().numpy()
            # # rot = rmat_to_euler(rot)
            # rot = rot[None, :]

            # action passed to env.step(action) is a numpy array
            # action = np.concatenate([pos, rot, gripper], axis=-1)[0]
            self.action_chunker.add_action(actions)
            action = self.action_chunker.get_action()

        return action


    def reset(self):
        self.obs_chunker.reset()
        self.action_chunker.reset()

    def enable_chaining(self):
        if hasattr(self.policy, "enable_chaining"):
            self.policy.enable_chaining()
        else:
            raise NotImplementedError("Chosen policy does not support chaining.")

class ActionChunker:
    """Wrapper for chunking actions. Takes in an action sequence; returns one action when queried.
    Returns None if already popped out all actions.
    """

    def __init__(self, n_act_steps):
        """
        Args:
            n_act_steps (int): number of actions to buffer before requiring a new action sequence to be added.
        """
        self.n_act_steps = n_act_steps
        self.actions = deque()

    def reset(self):
        self.action_history = None

    def add_action(self, action):
        """Add a sequence of actions to the chunker.

        Args:
            action (np.ndarray): An array of actions, shape (N, action_dim).
        """
        if not isinstance(action, np.ndarray):
            raise ValueError("Action must be a numpy array.")
        if len(action.shape) != 2:
            raise ValueError("Action array must have shape (N, action_dim).")

        # slice the actions into chunks of size n_act_steps
        action = action[:self.n_act_steps]

        # Extend the deque with the new actions
        self.actions.extend(action)

    def get_action(self):
        """Get the next action from the chunker.

        Returns:
            np.ndarray or None: The next action, or None if no actions are left.
        """
        if self.actions:
            return self.actions.popleft()
        else:
            return None


class ObsChunker:
    """
    Wrapper for chunking observations. Builds up a buffer of n_obs_steps observations and releases them all at once.
    """
    def __init__(self, n_obs_steps):
        """
        Args:
            n_obs_steps (int): number of observations to buffer before releasing them all at once.
        """
        self.n_obs_steps = n_obs_steps
        self.obs_history = None

    def reset(self):
        self.obs_history = None

    def add_obs(self, obs):
        if self.obs_history is None:
            self.obs_history = {}
            for k in obs:
                self.obs_history[k] = deque(maxlen=self.n_obs_steps)
        for k in obs:
            self.obs_history[k].append(obs[k])

    def get_obs_history(self):
        current_obs = {k: v[-1] for k, v in self.obs_history.items()} # Get the most recent observation
        while self.obs_history is None or len(next(iter(self.obs_history.values()))) < self.n_obs_steps:
            for k in current_obs:
                # add the current obs to the history
                self.obs_history[k].append(current_obs[k])

        obs_to_return = {k: np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history}
        return obs_to_return
