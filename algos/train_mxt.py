import sys
from matplotlib import pyplot as plt
import torch
import numpy as np
import os
import pickle
import argparse
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
import json
import wandb

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)

from algos.mxt_utils import load_data, compute_dict_mean, set_seed # data functions
from algos.policy import make_policy, make_optimizer

from locoman.config.config import Cfg
from locoman.config.go1_config import config_go1

import yaml
import gc

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)

def forward_pass(batch, policy):
    
    return policy.compute_loss(batch)

def train_bc(train_dataloader, val_dataloader, config, policy_config):
    import time
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)
    policy = make_policy(policy_class, policy_config)
    if config['load_pretrain']:
        loading_status = policy.deserialize(torch.load(f'{config["pretrained_path"]}', map_location='cuda'))# , eval=True) #'/policy_last.ckpt'
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    policy.cuda()
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params}")
    optimizer = make_optimizer(policy_class, policy)
    # if config['load_pretrain']:
    #     optimizer.load_state_dict(torch.load(f'{config["pretrained_path"]}/optimizer_last.ckpt', map_location='cuda'))
        

    min_val_loss = np.inf
    best_ckpt_info = None
    
    train_dataloader = repeater(train_dataloader)
    loss_list = []
    # l1_list = []
    # feature_loss_list = []
    val_loss_list = []
    tokenizer_params = list(policy.model.transformer_body.parameters())
    
    for step in tqdm(range(num_steps+1)):
        if step % validate_every == 0:
            print("training loss:", np.mean(loss_list) if len(loss_list) > 0 else "N/A")
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    del forward_dict
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['all_emb_loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)     
            if config['wandb']:       
                wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        summary_string += f'{k}_{kk}: {vv:.3f} '
                else:
                    summary_string += f'{k}: {v:.3f} '
            print(summary_string)
            val_loss_list.append(epoch_val_loss)
            del validation_dicts, validation_summary, epoch_val_loss
                
        # training
        policy.train() 
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict['all_emb_loss']
        loss.backward()
        loss_list.append(loss.item())
        
        optimizer.step()
        if config['wandb']:
            wandb.log(forward_dict, step=step) # not great, make training 1-2% slower
        
        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            #save optimizer state
            optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_step_{step}_seed_{seed}.ckpt')
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)
            # save plots
            plot_path = os.path.join(ckpt_dir, f'train_val_seed_{seed}.png')
            plt.figure()
            plt.plot(np.linspace(0, len(loss_list), len(loss_list)), loss_list, label='train')
            plt.plot(np.linspace(0, len(loss_list), len(val_loss_list)), val_loss_list, label='validation')
            # plt.ylim([-0.1, 8.0])
            plt.tight_layout()
            plt.legend()
            plt.title('loss curve')
            plt.savefig(plot_path)
            plt.close('all')
            plt.clf()
            print(f'Saved plots to {ckpt_dir}')
        if step % 2000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_last.ckpt')
            torch.save(optimizer.state_dict(), optimizer_ckpt_path)
            gc.collect()
            
        del forward_dict, loss, data
            
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)
    optimizer_ckpt_path = os.path.join(ckpt_dir, f'optimizer_last.ckpt')
    torch.save(optimizer.state_dict(), optimizer_ckpt_path)

    return best_ckpt_info

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1

def main_train(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    dataset_dir = args['dataset_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']
    backbone = args['backbone']
    same_backbones = args['same_backbones']
    use_wrist = args['use_wrist']
    train_ratio = args['train_ratio']
    min_val_num = args['min_val_num']
    
    ckpt_dir = f'{ckpt_dir}_{task_name}_{policy_class}'
    if os.path.isdir(ckpt_dir):
        print(f'ckpt_dir {ckpt_dir} already exists, exiting')
        return
    args['ckpt_dir'] = ckpt_dir 
    # get task parameters
    is_sim = task_name[:4] == 'sim_'

    with open(args['embodiment_config_path'], 'r') as f:
        emb_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(args['trunk_config_path'], 'r') as f:
        transformer_dict = yaml.load(f, Loader=yaml.FullLoader)
        
    # override action horizon in action head configs with chunk_size
    for key in emb_dict.keys():
        if key == 'human' or key == 'locoman':
            for modality in emb_dict[key]['action_head'].keys():
                if 'action_horizon' in emb_dict[key]['action_head'][modality]:
                    emb_dict[key]['action_head'][modality]['action_horizon'] = args['chunk_size']
    
    name_filter = lambda _: True

    print(f'===========================START===========================:')
    print(f"{task_name}")
    print(f'===========================Config===========================:')
    print(f'ckpt_dir: {ckpt_dir}')
    print(f'policy_class: {policy_class}')
    # TODO: load multiple datasets from different embodiments
    train_dataloader, val_dataloader, embodiment_stats, embodiment_list, _ = load_data(dataset_dir, name_filter, emb_dict, 
                                                                        batch_size_train, batch_size_val, args['chunk_size'], 
                                                                        args['skip_mirrored_data'], args['load_pretrain'], 
                                                                        policy_class, # stats_dir_l=stats_dir, 
                                                                        # sample_weights=sample_weights, 
                                                                        train_ratio=train_ratio,
                                                                        width=args['width'],
                                                                        height=args['height'],
                                                                        normalize_resnet=args['normalize_resnet'],
                                                                        data_aug=args['data_aug'],
                                                                        observation_name=['qpos'],
                                                                        feature_loss = args['feature_loss_weight'] > 0,
                                                                        grayscale = args['grayscale'],
                                                                        randomize_color = args['randomize_color'],
                                                                        randomize_data_degree = args['randomize_data_degree'],
                                                                        randomize_data = args['randomize_data'],
                                                                        # randomize_index = randomize_index,
                                                                        aggregate_image=False,
                                                                        min_val_num=min_val_num,
                                                                        agg_modalities=args['agg_modalities'],
                                                            )
    lr_backbone = 1e-5
    backbone = args['backbone']
        
    policy_config = {
        'lr': args['lr'],
        'lr_backbone': lr_backbone,
        'backbone': backbone,
        'norm_stats': embodiment_stats,
        "embodiment_args_dict": emb_dict,
        "transformer_args": transformer_dict,
        "share_cross_attn": args['share_cross_attn'],
        "agg_modalities": args['agg_modalities'], 
        }
    
    print(f'====================FINISH INIT POLICY========================:')
    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        # 'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': False,
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],
        'height':args['height'],
        'width':args['width'],
        'normalize_resnet': args['normalize_resnet'],
        'wandb': args['wandb'],
        'pretrained_path': args['pretrained_path'],
        'randomize_data_degree': args['randomize_data_degree'],
        'randomize_data': args['randomize_data'],
    }
    all_configs = {**config, **args, 'policy_config': {k: v for k, v in policy_config.items() if k != 'norm_stats'}}
    print(all_configs)
    
    if args['width'] < 0:
        args['width'] = None
    if args['height'] < 0:
        args['height'] = None

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    all_config_path = os.path.join(ckpt_dir, 'all_configs.json')
    expr_name = ckpt_dir.split('/')[-1]
    if not is_eval and args['wandb']:
        wandb.init(project=PROJECT_NAME, reinit=True, name=expr_name) # , entity=WANDB_USERNAME
        wandb.config.update(config)
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    with open(all_config_path, 'w') as fp:
        json.dump(all_configs, fp, indent=4)

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(embodiment_stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, policy_config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    if args['wandb']:
        wandb.finish()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action='store', type=str, help='config file', required=False)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', default="h1_ckpt/0", action='store', type=str, help='ckpt_dir')
    parser.add_argument('--policy_class', default="cross", action='store', type=str, help='policy_class, capitalize')
    parser.add_argument('--task_name', default="sim_pushcube", action='store', type=str, help='task_name')
    # parser.add_argument('--encoder_path', default=None, action='store', type=str, help='encoder_path')
    
    parser.add_argument('--batch_size', default=32, action='store', type=int, help='batch_size')
    parser.add_argument('--seed', default=0, action='store', type=int, help='seed')
    parser.add_argument('--num_steps', default=20000, action='store', type=int, help='num_steps')
    
    parser.add_argument('--lr', default=1e-5, action='store', type=float, help='lr')
    parser.add_argument('--lr_tokenizer', default=1e-5, action='store', type=float, help='lr_tokenizer')
    parser.add_argument('--lr_action_head', default=1e-5, action='store', type=float, help='lr_actino_head')
    parser.add_argument('--lr_trunk', default=1e-5, action='store', type=float, help='lr_trunk')
    
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--pretrained_path', action='store', type=str, help='pretrained_path', required=False)
    
    parser.add_argument('--eval_every', action='store', type=int, default=100000, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--no_encoder', action='store_true')
    # dec_layers
    parser.add_argument('--dec_layers', action='store', type=int, default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, default=8, required=False)
    parser.add_argument('--use_pos_embd_image', action='store', type=int, default=1, required=False)
    parser.add_argument('--use_pos_embd_action', action='store', type=int, default=1, required=False)
    
    # feature_loss_weight
    parser.add_argument('--feature_loss_weight', action='store', type=float, default=0.0)
    # self_attention
    parser.add_argument('--self_attention', action="store", type=int, default=1)
    
    # for backbone 
    parser.add_argument('--backbone', type=str, default='dinov2')
    parser.add_argument('--same_backbones', action='store_true')
    # use mask
    parser.add_argument('--use_mask', action='store_true')
    # use image from wrist camera
    parser.add_argument('--use_wrist', action='store_true')
    
    # for image 
    parser.add_argument('--width', type=int, default=660)
    parser.add_argument('--height', type=int, default=420)
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--normalize_resnet', action='store_true') ### not used - always normalize - in the model.forward
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--randomize_color', action='store_true')
    parser.add_argument('--randomize_data', action='store_true')
    parser.add_argument('--randomize_data_degree', action='store', type=int, default=3)
    parser.add_argument('--pretrain_image_width', action='store', type=int, default=1276)
    parser.add_argument('--pretrain_image_height', action='store', type=int, default=480)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--model_type', type=str, default="HIT")
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # dataset
    parser.add_argument('--dataset_dir', default="demonstrations", action='store', type=str, help='the folder where the dataset is placed')
    parser.add_argument('--train_ratio', default=0.99, action='store', type=float, help='train_ratio')
    parser.add_argument('--min_val_num', default=1, action='store', type=int, help='min_val_num')
    
    # mxt config
    parser.add_argument('--embodiment_config_path', default="embodiments.yaml", action='store', type=str, help='embodiment_config_path')
    parser.add_argument('--trunk_config_path', default="transformer_trunk.yaml", action='store', type=str, help='trunk_config_path')
    parser.add_argument('--share_cross_attn', action='store_true')
    
    parser.add_argument('--wandb_name', action='store', type=str, help='wandb_name', default='Human2Locoman')
        
    parser.add_argument('--agg_modalities', action='store_true', default=False)
    
    args = vars(parser.parse_args())
    torch.cuda.set_device(args['gpu_id'])
    PROJECT_NAME = args['wandb_name'] # 'Human2Locoman'
    WANDB_USERNAME = "yarun"
    
    config_go1(Cfg)
    
    main_train(args)
    