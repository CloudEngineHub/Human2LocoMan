"""
Utility functions. Code adapted from HPT.
"""

from typing import List
import cv2
import h5py
import torch.nn.functional as F
import torch.nn as nn
import torch

import os
from PIL import Image
import numpy as np


import gc
import einops
import json
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from algos.utils import find_all_hdf5, BatchSampler


# global model cache
global_vision_model = None  # to be assigned
global_language_model = None  # to be assigned
global_vision_processor = None # to be assigned
global_language_processor = None # to be assigned


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_args_json(path, args, convert_to_vars=False):
    """
    Save the arguments to a JSON file.

    Args:
        path (str): The path to save the JSON file.
        args (object): The arguments to be saved.
        convert_to_vars (bool, optional): Whether to convert the arguments to variables. Defaults to False.
    """
    mkdir_if_missing(path)
    arg_json = os.path.join(path, "config.json")
    with open(arg_json, "w") as f:
        if convert_to_vars:
            args = vars(args)
        json.dump(args, f, indent=4, sort_keys=True)


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)


def mkdir_if_missing(dst_dir):
    """make destination folder if it's missing"""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

def get_optimizer(
    optimizer_spec,
    policy,
    optimizer_extra=None,
    **kwargs,
):
    """
    initializer network optimizer
    """
    trunk_params = [v for k, v in policy.named_parameters() if "trunk" in k]
    nontrunk_params = [v for k, v in policy.named_parameters() if "trunk" not in k]
    params = [
        {"params": trunk_params},
        {"params": nontrunk_params, "lr": optimizer_spec.lr * optimizer_extra.nontrunk_lr_scale},
    ]

    opt_i = eval(optimizer_spec["_target_"])(
        params=params,
        **{k: v for k, v in optimizer_spec.items() if k != "_target_"},
    )
    return opt_i


def dict_apply(x, func):
    """
    Apply a function to all values in a dictionary recursively.

    Args:
        x (dict or any): The dictionary or value to apply the function to.
        func (function): The function to apply.

    Returns:
        dict or any: The resulting dictionary or value after applying the function.
    """
    dict_type = type(x)
    if type(x) is not dict_type:
        return func(x)

    result = dict_type()
    for key, value in x.items():
        if isinstance(value, dict_type):
            result[key] = dict_apply(value, func)
        else:
            try:
                result[key] = func(value)
            except:
                result[key] = value  # fallback
    return result


def compute_dict_mean(epoch_dicts):
    """
    Compute the mean of values in a list of dictionaries that may have different structures.
    For each leaf node (non-dictionary value), computes the mean of all available values across dicts.
    
    Args:
        epoch_dicts: List of dictionaries that may have different structures
        
    Returns:
        Dictionary containing mean values for all leaf nodes that appear in any input dict
    """
    if not epoch_dicts:
        return {}
    
    # Collect all unique keys across all dictionaries
    all_keys = set()
    for d in epoch_dicts:
        all_keys.update(d.keys())
    
    result = {}
    for k in all_keys:
        # Collect all values for this key
        values = []
        for d in epoch_dicts:
            if k in d:
                if isinstance(d[k], dict):
                    # For nested dictionaries, collect all corresponding dicts
                    values.append(d[k])
                else:
                    # For leaf values, include in the mean calculation
                    values.append(d[k])
        
        if not values:
            continue
        # If the values are dictionaries, recurse
        if all(isinstance(v, dict) for v in values):
            result[k] = compute_dict_mean(values)
        # If we have any numeric values, compute their mean
        elif any(isinstance(v.item(), (int, float)) for v in values):
            numeric_values = [v.item() for v in values if isinstance(v.item(), (int, float))]
            if numeric_values:
                result[k] = sum(numeric_values) / len(numeric_values)
                
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    # set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_sinusoid_encoding_table(position_start, position_end, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(position_start, position_end)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def unnormalize_image_numpy(image):
    """
    Unnormalizes an image in numpy format.

    Args:
        image (numpy.ndarray): The input image in numpy format.

    Returns:
        numpy.ndarray: The unnormalized image.

    Shape:
        - Input: (C, H, W)
        - Output: (C, H, W)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    image = image * std + mean
    image = image * 255
    image = image.astype(np.uint8)
    return image


def normalize_image_numpy(image: np.ndarray, resize: bool = True) -> np.ndarray:
    """
    Normalize an image in numpy format.

    Args:
        image (numpy.ndarray): The input image in H x W x 3 (uint8) format.

    Returns:
        numpy.ndarray: The normalized image in 3 x H x W format.

    Notes:
        - The input image is resized to (224, 224) using cv2.resize.
        - The image is normalized using the mean and standard deviation values from the ImageNet dataset.
        - The resulting image is transposed to have dimensions of 3 x H x W.
    """
    if resize:
        image = cv2.resize(image, (224, 224))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image / 255.0

    # convert to array
    image = np.asarray(image)

    # normalize
    image = (image - mean) / std
    return image.transpose(2, 0, 1)


def dict_apply_device(x, device):
    """
    Apply the specified device to all tensors in a nested dictionary.

    Args:
        x (dict): The nested dictionary to apply the device to.
        device (torch.device): The device to apply.

    Returns:
        dict: The nested dictionary with tensors moved to the specified device.
    """
    if type(x) is not dict:
        return value.to(device)

    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply_device(value, device)
        else:
            result[key] = value.to(device)
    return result

def download_from_huggingface(huggingface_repo_id: str):
    import huggingface_hub

    folder = huggingface_hub.snapshot_download(huggingface_repo_id)
    return folder


def load_data(dataset_dir_l, name_filter, emb_dict, batch_size_train, batch_size_val, 
              chunk_size, skip_mirrored_data=False, load_pretrain=False, 
              policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.95,
              width=None, height=None, normalize_resnet=False, data_aug=False, observation_name=None,
              feature_loss=False, grayscale=False, randomize_color=False,
              randomize_index=None,randomize_data_degree=0,randomize_data=False,aggregate_image=True, min_val_num=1, agg_modalities=False):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    print('dataset_path_list_list', dataset_path_list_list)
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    dataset_path_list = sorted(dataset_path_list)
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    shuffled_episode_ids_0 = np.arange(num_episodes_0)
    val_episode_num_0 = max(min_val_num, int((1-train_ratio) * num_episodes_0))
    train_episode_ids_0 = shuffled_episode_ids_0[:-val_episode_num_0]
    val_episode_ids_0 = shuffled_episode_ids_0[-val_episode_num_0:]
    print(f'train_episode_ids_0: {train_episode_ids_0}')
    print(f'val_episode_ids_0: {val_episode_ids_0}')
    
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')
    print("test on ",dataset_path_list[val_episode_ids_l[0][0]])
    print(f'validation episodes: {[dataset_path_list[idx] for idx in val_episode_ids_0]}')

    _, all_episode_len, embodiment_list = get_norm_stats(dataset_path_list, embodiment_dict=emb_dict)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    print('train_episode_len_l', train_episode_len_l)
    print('val_episode_len_l', val_episode_len_l)
    print('train_episode_len', train_episode_len)
    print('val_episode_len', val_episode_len)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    embodiment_norm_stats, _, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]),
                                   embodiment_dict=emb_dict)
    print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, embodiment_norm_stats, train_episode_ids, train_episode_len, chunk_size, 
                                    policy_class, width=width, height=height,normalize_resnet=normalize_resnet,data_aug=data_aug,
                                    feature_loss=feature_loss,grayscale=grayscale,randomize_color=randomize_color,
                                    randomize_index=randomize_index,randomize_data_degree=randomize_data_degree,randomize_data=randomize_data,
                                    aggregate_image=aggregate_image, agg_modalities=agg_modalities)
    val_dataset = EpisodicDataset(dataset_path_list, embodiment_norm_stats, val_episode_ids, val_episode_len, chunk_size, 
                                  policy_class, width=width, height=height,normalize_resnet=normalize_resnet,data_aug=False,
                                  feature_loss=feature_loss,grayscale=grayscale,randomize_color=False,
                                  aggregate_image=aggregate_image, agg_modalities=agg_modalities)
    train_num_workers = 8 if train_dataset.data_aug else 8
    val_num_workers = 8 if train_dataset.data_aug else 8
    print(f'Augment images: {train_dataset.data_aug}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    
    def collate_fn(samples):
        def aggregate_dict(data: list):
            if isinstance(data[0], dict):
                return {key: aggregate_dict([sample[key] for sample in data]) for key in data[0].keys()}
            else:
                return torch.stack(data)
            
        embodiment_array = np.array([sample[1] for sample in samples], dtype=np.str_)
        input_dict = dict()
        for emb in np.unique(embodiment_array):
            data_emb_list = [sample[0] for sample in samples if sample[1] == emb]
            input_dict[emb] = aggregate_dict(data_emb_list)
        
        return input_dict
        
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, collate_fn=collate_fn, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, collate_fn=collate_fn, prefetch_factor=2)

    return train_dataloader, val_dataloader, embodiment_norm_stats, embodiment_list, train_dataset.is_sim

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, norm_stats, episode_ids, episode_len, 
                 chunk_size, policy_class,width=None, height=None, normalize_resnet=False,data_aug=False,
                 feature_loss=False,grayscale=False,randomize_color=False,
                 randomize_index=None,randomize_data_degree=0,randomize_data=False, aggregate_image=True,
                 agg_modalities=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = np.array(dataset_path_list, dtype=np.string_)
        self.obs_keys = dict()
        self.action_keys = dict()
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        self.transformations = None
        self.width = width
        self.height = height
        self.data_aug = data_aug
        self.feature_loss = feature_loss
        self.randomize_data = randomize_data
        self.randomize_data_radian = randomize_data_degree/180*np.pi
        self.randomize_index = randomize_index
        self.grayscale = grayscale
        self.randomize_color = randomize_color
        self.aggregate_image = aggregate_image
        self.agg_modalities = agg_modalities
        if self.data_aug:
            #has nothing to do with the deployment of the model 
            self.transformations = [
                transforms.ColorJitter(hue=0.5,saturation=0.5),
                # transforms.Pad(padding=[int(self.width * 0.05), int(self.height * 0.05)], padding_mode='edge'),
                # transforms.RandomCrop(size=[self.height,self.width])]
            ]
        else:
            self.transformations = None
  
        self.normalize_resnet = normalize_resnet
        if self.normalize_resnet:
            #need to normalize the image to the same mean and std as the resnet model during depolyment
            self.normalize_resnet_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        for embodiment in norm_stats.keys():
            self.obs_keys[embodiment] = norm_stats[embodiment]['obs'].keys()
            self.action_keys[embodiment] = norm_stats[embodiment]['action'].keys()
        
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = str(self.dataset_path_list[episode_id], encoding='utf-8')
        if "flipped" in dataset_path:
            flipped_data = True
        else:
            flipped_data = False
        with h5py.File(dataset_path, 'r') as root:
            is_sim = False
            compressed = root.attrs.get('compress', False)
            embodiment = root.attrs.get('embodiment', 'locoman')
            episode_len = get_episode_len(root)
            image_dict = dict()
            image_mask_dict = dict()
            if self.feature_loss:
                image_dict_future = dict()
            proprio_dict = dict()
            proprio_mask_dict = dict()
            action_dict = dict()
            action_mask_dict = dict()
            for key in self.obs_keys[embodiment]:
                if 'image' in key:
                    image_dict[key] = root[KEY_PATH_MAP[key]][start_ts] # will convert to tensor after preprocess
                    if key in KEY_MASK_PATH_MAP:
                        image_mask_dict[key] = torch.tensor(root[KEY_MASK_PATH_MAP[key]][0])
                    else:
                        image_mask_dict[key] = torch.tensor([True])
                    if self.feature_loss:
                        dummy_index = min(start_ts+self.chunk_size, episode_len - 1)
                        image_dict_future[key] = root[KEY_PATH_MAP[key]][dummy_index]
                elif 'state' in key:
                    try:
                        proprio_dict[key] = torch.from_numpy(root[KEY_PATH_MAP[key]][start_ts]).float()
                    except Exception as e:
                        proprio_dict[key] = torch.from_numpy(root[KEY_PATH_MAP_ALT[key]][start_ts]).float()
                    if key in KEY_MASK_PATH_MAP:
                        proprio_mask_dict[key] = torch.tensor(root[KEY_MASK_PATH_MAP[key]])
                    else:
                        proprio_mask_dict[key] = torch.tensor([True] * proprio_dict[key].shape[-1])
            for key in self.action_keys[embodiment]:
                if is_sim:
                    action_dict[key] = torch.from_numpy(root[KEY_PATH_MAP[key]][start_ts:])
                else:
                    action_dict[key] = torch.from_numpy(root[KEY_PATH_MAP[key]][max(0, start_ts - 1):])
                if key in KEY_MASK_PATH_MAP:
                    action_mask_dict[key] = torch.tensor(root[KEY_MASK_PATH_MAP[key]])
                else:
                    action_mask_dict[key] = torch.tensor([True] * action_dict[key].shape[-1])
                    
            
            if compressed:
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    if self.width is not None and self.height is not None: 
                        decompressed_image = cv2.resize(decompressed_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    image_dict[cam_name] = np.array(decompressed_image)
                if self.feature_loss:
                    for cam_name in image_dict_future.keys():
                        decompressed_image = cv2.imdecode(image_dict_future[cam_name], 1)
                        if self.width is not None and self.height is not None: 
                            decompressed_image = cv2.resize(decompressed_image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                        image_dict_future[cam_name] = np.array(decompressed_image)
                        
            # get all actions after and including start_ts
            if is_sim:
                action_len = episode_len - start_ts
            else:
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        is_pad = np.zeros(self.max_episode_len)
        is_pad[action_len:] = 1
        is_pad = is_pad[:self.chunk_size]
        is_pad = torch.from_numpy(is_pad).bool()
        for key in action_dict.keys():
            orig_action = action_dict[key]
            action_dim = orig_action.shape[-1]
            padded_action = np.zeros((self.max_episode_len, action_dim), dtype=np.float32)
            padded_action[:action_len] = orig_action
            padded_action = padded_action[:self.chunk_size]
            action_dict[key] = torch.from_numpy(padded_action).float()

        order = np.random.permutation(3) if self.randomize_color else None
        for key in image_dict.keys():
            image_tensor = torch.from_numpy(image_dict[key]).float()
            if self.feature_loss:
                future_image_tensor = torch.from_numpy(image_dict_future[key]).float()
                image_tensor = torch.cat([image_tensor, future_image_tensor], dim=0)
                del future_image_tensor, image_dict_future[key]
            # image_tensor = image_tensor / 255.0
            image_tensor.div_(255.0)
            image_tensor = torch.einsum('h w c -> c h w', image_tensor)
            if flipped_data:
                image_tensor = image_tensor.flip(-1)
            if self.randomize_color:
                image_tensor = image_tensor[order,:,:]
            if self.grayscale:
                image_tensor = torch.mean(image_tensor, dim=0, keepdim=True).repeat(3,1,1)
            if self.data_aug:
                for transform in self.transformations:
                    image_tensor = transform(image_tensor)
            image_dict[key] = image_tensor
            del image_tensor

        if self.agg_modalities:
            main_image = image_dict['main_image']
            main_image_mask = image_mask_dict['main_image']
            c, h, w = main_image.shape
            main_image_left = main_image[:, :, :w//2]
            main_image_right = main_image[:, :, w//2:]
            if 'wrist_image' in image_dict:     
                wrist_image = image_dict['wrist_image']
                wrist_image_mask = image_mask_dict['wrist_image']
                all_images = torch.concat([main_image_left, main_image_right, wrist_image], dim=-1)
                all_images_mask = wrist_image_mask
            else:
                all_images = torch.concat([main_image_left, main_image_right], dim=-1)
                all_images_mask = main_image_mask
            image_dict = {'all_images': all_images}
            image_mask_dict = {'all_images': all_images_mask}
            proprio_keys = [key for key in proprio_dict.keys()]
            all_proprio = torch.concat([proprio_dict[key] for key in proprio_dict.keys()], dim=0)
            proprio_mask_keys = [key for key in proprio_mask_dict.keys()]
            all_proprio_mask = torch.concat([proprio_mask_dict[key] for key in proprio_mask_dict.keys()], dim=0)
            proprio_dict = {'all_proprio_states': all_proprio}
            proprio_mask_dict = {'all_proprio_states': all_proprio_mask}
            all_actions = torch.concat([action_dict[key] for key in action_dict.keys()], dim=-1)
            all_actions_mask = torch.concat([action_mask_dict[key] for key in action_mask_dict.keys()], dim=0)
            action_dict = {'all_actions': all_actions}
            action_mask_dict = {'all_actions': all_actions_mask}
        
        ret_dict = {
            'obs': {**image_dict, **proprio_dict},
            'obs_mask': {**image_mask_dict, **proprio_mask_dict},
            'action': action_dict,
            'action_mask': action_mask_dict,
            'is_pad': is_pad,
        }
        
        del image_dict, image_mask_dict, proprio_dict, proprio_mask_dict, action_dict, action_mask_dict, is_pad
        
        return ret_dict, embodiment


def parse_embodiment_keys(emb_dict:dict, embodiment:str):
    obs_keys = []
    action_keys = []
    for key in emb_dict[embodiment]['tokenizer'].keys():
        if 'state' in key or 'image' in key:  # do not normalize images based on data
            obs_keys.append(key)
    for key in emb_dict[embodiment]['action_head'].keys():
        action_keys.append(key)
    return obs_keys, action_keys


def get_norm_stats(dataset_path_list, embodiment_dict):
    all_obs_data = dict()
    all_action_data = dict()
    all_episode_len = []
    # obs_stats = dict()
    # action_stats = dict()
    # embodiment = None
    all_embodiment_stats = dict()
    embodiment_list = []
    total_emb_episode_len = dict()

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                embodiment = root.attrs.get('embodiment', 'locoman')
                if embodiment not in embodiment_list:
                    embodiment_list.append(embodiment)
                    print('---------discovered embodiment', embodiment, '----------')
                    obs_entry_keys, action_entry_keys = parse_embodiment_keys(embodiment_dict, embodiment)
                    all_obs_data[embodiment] = {key: [] for key in obs_entry_keys}
                    all_action_data[embodiment] = {key: [] for key in action_entry_keys}
                    # all_episode_len_dict[embodiment] = []
                    all_embodiment_stats[embodiment] = dict()
                    total_emb_episode_len[embodiment] = 0
                for key in obs_entry_keys:
                    if 'image' not in key: # don't get mean and std from image data
                        try:
                            all_obs_data[embodiment][key].append(root[KEY_PATH_MAP[key]][()])
                        except Exception as e:
                            print(key)
                            all_obs_data[embodiment][key].append(root[KEY_PATH_MAP_ALT[key]][()])
                    # else:
                    #     obs_stats[embodiment][key] = {"mean": None, "std": None}
                for key in all_action_data[embodiment].keys():
                    all_action_data[embodiment][key].append(root[KEY_PATH_MAP[key]][()])
                all_episode_len.append(len(root[KEY_PATH_MAP[obs_entry_keys[0]]][()]))
                total_emb_episode_len[embodiment] += all_episode_len[-1]
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            raise e

    for embodiment in embodiment_list:
        obs_stats = dict()
        action_stats = dict()
        for key in all_obs_data[embodiment].keys():
            if 'image' not in key:
                all_obs_data[embodiment][key] = np.concatenate(all_obs_data[embodiment][key], axis=0)
            else:
                obs_stats[key] = {"mean": None, "std": None}
        for key in all_action_data[embodiment].keys():
            all_action_data[embodiment][key] = np.concatenate(all_action_data[embodiment][key], axis=0)
            
        obs_stats.update({key: {"mean": np.mean(all_obs_data[embodiment][key], axis=0), "std": np.std(all_obs_data[embodiment][key], axis=0)} for key in all_obs_data[embodiment].keys() if 'image' not in key})
        action_stats.update({key: {"mean": np.mean(all_action_data[embodiment][key], axis=0), "std": np.std(all_action_data[embodiment][key], axis=0)} for key in all_action_data[embodiment].keys()})
        all_embodiment_stats[embodiment] = {'obs': obs_stats, 'action': action_stats}

    del all_obs_data, all_action_data
    
    for embodiment in total_emb_episode_len.keys():
        print(f'-------------Embodiment {embodiment} has {total_emb_episode_len[embodiment]} steps--------------------')
    
    return all_embodiment_stats, all_episode_len, embodiment_list

KEY_PATH_MAP = {
    'main_image': '/observations/images/main',
    'wrist_image': '/observations/images/wrist',
    'body_pose_state': '/observations/proprioceptions/body',
    'eef_pose_state': '/observations/proprioceptions/eef',
    'eef_to_body_pose_state': '/observations/proprioceptions/eef_to_body',
    'gripper_state': '/observations/proprioceptions/gripper',
    # 'other_proprio_state_qpos': '/observations/proprioceptions/other/joint_pos',
    # 'other_proprio_state_qvel': '/observations/proprioceptions/other/joint_vel',
    # 'other_proprio_left_hand': '/observations/proprioceptions/other/left_hand_joints',
    # 'other_proprio_right_hand': '/observations/proprioceptions/other/right_hand_joints',
    'delta_eef_pose': '/actions/delta_eef',
    'delta_body_pose': '/actions/delta_body',
    'delta_gripper': '/actions/delta_gripper',
    'eef_pose': '/actions/eef',
    'body_pose': '/actions/body',
    'gripper': '/actions/gripper',
}

KEY_PATH_MAP_ALT = {
    'main_image': '/observations/images/main',
    'wrist_image': '/observations/images/wrist',
    'body_pose_state': '/observations/proprioceptions/body',
    'eef_pose_state': '/observations/proprioceptions/eef',
    'eef_to_body_pose_state': '/observations/proprioceptions/relative',
    'gripper_state': '/observations/proprioceptions/gripper',
    # 'other_proprio_state_qpos': '/observations/proprioceptions/other/joint_pos',
    # 'other_proprio_state_qvel': '/observations/proprioceptions/other/joint_vel',
    # 'other_proprio_left_hand': '/observations/proprioceptions/other/left_hand_joints',
    # 'other_proprio_right_hand': '/observations/proprioceptions/other/right_hand_joints',
    'delta_eef_pose': '/actions/delta_eef',
    'delta_body_pose': '/actions/delta_body',
    'delta_gripper': '/actions/delta_gripper',
    'eef_pose': '/actions/eef',
    'body_pose': '/actions/body',
    'gripper': '/actions/gripper',
}

# missing keys default to true
KEY_MASK_PATH_MAP = {
    'main_image': '/masks/img_main',
    'wrist_image': '/masks/img_wrist',
    'body_pose_state': '/masks/proprio_body',
    'eef_pose_state': '/masks/proprio_eef',
    'eef_to_body_pose_state': '/masks/proprio_eef',
    'gripper_state': '/masks/proprio_gripper',
    'delta_eef_pose': '/masks/act_eef',
    'delta_body_pose': '/masks/act_body',
    'delta_gripper': '/masks/act_gripper',
    'eef_pose': '/masks/act_eef',
    'body_pose': '/masks/act_body',
    'gripper': '/masks/act_gripper',
}

def get_episode_len(hdfile):
    for key in hdfile.keys():
        if isinstance(hdfile[key], h5py.Group):
            res = get_episode_len(hdfile[key])
            return res
        else:
            return hdfile[key][()].shape[0]