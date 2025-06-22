# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import argparse
from typing import List
import einops
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer, build_transformer_decoder
from algos.detr.models.mxt_definitions.tokenizer import ImageTokenizer, ProprioTokenizer
from algos.detr.models.mxt_definitions.action_head import MLP, Conv1D, TransformerDecoderHead, GMMHead

import numpy as np

import IPython
e = IPython.embed

class SimpleNormalizer(nn.Module):
    def __init__(self, mean, std):
        super(SimpleNormalizer, self).__init__()
        std = torch.maximum(std, torch.tensor(1e-6))
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x: torch.Tensor):
        return (x - self.mean) / self.std
    
    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor):
        return x * self.std + self.mean
            
def dict_to_device(data, device):
    new_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            new_data[key] = dict_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            new_data[key] = value.to(device)
        else:
            new_data[key] = value
    return new_data

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) # 1, n_position, d_hidden


class CrossTransformer(nn.Module):
    ''' The cross transformer model, drawing from HPT and CrossFormer'''
    def __init__(
        self,
        embodiment_args_dict: dict = None,
        transformer_args: dict = None,
        norm_stats: dict = None,
        agg_modalities: bool = False,
        **kwargs,
        ):
        super().__init__()
        self.tokenizers = nn.ModuleDict()
        self.action_heads = nn.ModuleDict()
        self.normalizers = nn.ModuleDict()
        self.query_len = dict()
        self.embodiment_args_dict = embodiment_args_dict
        self.transformer_args = transformer_args
        self.token_postprocessing = transformer_args["token_postprocessing"]
        self.agg_modalities = agg_modalities
        for embodiment in [emb for emb in embodiment_args_dict.keys() if emb in norm_stats.keys()]:
            if isinstance(embodiment_args_dict[embodiment], dict):
                self.tokenizers[embodiment] = self._build_tokenizer(embodiment_args_dict[embodiment]["tokenizer"]) # return a dictionary of tokenizers
                self.action_heads[embodiment] = self._build_action_head(embodiment_args_dict[embodiment]["action_head"])
                self.normalizers[embodiment] = self._build_normalizers(embodiment_args_dict[embodiment], norm_stats[embodiment])
        self._build_transformer_body(transformer_args)
        self.query_len = {modality: transformer_args["output_len"][modality] for modality in transformer_args["output_len"].keys()}
        
    def _build_tokenizer(self, args):
        if not self.agg_modalities:
            tokenizers = nn.ModuleDict()
            for modality in args.keys():
                if "image" in modality:
                    tokenizers[modality] = ImageTokenizer(**args[modality])
                    tokenizers[modality].init_cross_attn(args, modality)
                elif "state" in modality:
                    tokenizers[modality] = ProprioTokenizer(**args[modality])
                    tokenizers[modality].init_cross_attn(args, modality)
                # else:
                #     raise NotImplementedError
            total_params_tokenizers = sum([sum([p.numel() for p in tokenizers[modality].parameters() if p.requires_grad]) for modality in tokenizers.keys()])
            print(f"Total parameters in tokenizers: {total_params_tokenizers}")
            return tokenizers
        else:
            tokenizers = nn.ModuleDict()
            image_arg_list = [args[modality] for modality in args.keys() if "image" in modality]
            state_arg_list = [args[modality] for modality in args.keys() if "state" in modality]
            agg_image_arg = {'output_dim': image_arg_list[0]['output_dim'],
                             'weights': image_arg_list[0]['weights'],
                             'resnet_model': image_arg_list[0]['resnet_model'],
                             'num_of_copy': image_arg_list[0]['num_of_copy'],
                             'token_num': sum([arg['token_num'] for arg in image_arg_list]),
                             }
            tokenizers['all_images'] = ImageTokenizer(**agg_image_arg)
            agg_proprio_arg = {'output_dim': state_arg_list[0]['output_dim'],
                               'widths': state_arg_list[0]['widths'],
                               'tanh_end': state_arg_list[0]['tanh_end'],
                               'ln': state_arg_list[0]['ln'],
                               'input_dim': sum([arg['input_dim'] for arg in state_arg_list]),
                               'token_num': sum([arg['token_num'] for arg in state_arg_list]),
                               }
            tokenizers['all_proprio_states'] = ProprioTokenizer(**agg_proprio_arg)
            new_args = args.copy()
            new_args['all_images'] = agg_image_arg
            new_args['all_proprio_states'] = agg_proprio_arg
            tokenizers['all_images'].init_cross_attn(new_args, 'all_images')
            tokenizers['all_proprio_states'].init_cross_attn(new_args, 'all_proprio_states')
            return tokenizers

    def _build_transformer_body(self, args, normalizer=None):
        self.trunk_mode = args["trunk_mode"]
        args_obj = argparse.Namespace()
        args_obj.__dict__.update(args)
        if self.trunk_mode == "encoder_decoder":
            self.transformer_body = build_transformer(args_obj)
            total_params_trunk = sum([p.numel() for p in self.transformer_body.parameters() if p.requires_grad])
            print(f"Total parameters in transformer body: {total_params_trunk}")
        elif self.trunk_mode == "decoder_only":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _build_action_head(self, args):
        if not self.agg_modalities:
            action_heads = nn.ModuleDict()
            # normalizers = nn.ModuleDict()
            for modality in args.keys():
                if args[modality]["type"] == "mlp" :
                    action_heads[modality] = MLP(**args[modality])
                elif args[modality]["type"] == "conv1d":
                    action_heads[modality] = Conv1D(**args[modality])
                elif args[modality]["type"] == "transformer":
                    action_heads[modality] = TransformerDecoderHead(**args[modality])
                elif args[modality]["type"] == "gmm":
                    action_heads[modality] = GMMHead(**args[modality])
                else:
                    raise NotImplementedError
                
            total_params_action_heads = sum([sum([p.numel() for p in action_heads[modality].parameters() if p.requires_grad]) for modality in action_heads.keys()])
            print(f"Total parameters in action heads: {total_params_action_heads}")
            return action_heads
        
        else:
            action_heads = nn.ModuleDict()
            arg_list = [args[modality] for modality in args.keys()]
            assert arg_list[0]['type'] == 'transformer', "only transformer is supported for now"
            agg_arg = {'type': arg_list[0]['type'],
                       'input_dim': arg_list[0]['input_dim'],
                       'output_dim': sum([arg['output_dim'] for arg in arg_list]),
                       'crossattn_modality_dropout': arg_list[0]['crossattn_modality_dropout'],
                       'crossattn_heads': arg_list[0]['crossattn_heads'],
                       'crossattn_dim_head': arg_list[0]['crossattn_dim_head'],
                       'action_horizon': arg_list[0]['action_horizon'],
                       }
            if agg_arg['type'] == "mlp" :
                action_heads['all_actions'] = MLP(**agg_arg)
            elif agg_arg['type'] == "conv1d":
                action_heads['all_actions'] = Conv1D(**agg_arg)
            elif agg_arg['type'] == "transformer":
                action_heads['all_actions'] = TransformerDecoderHead(**agg_arg)
            elif agg_arg['type'] == "gmm":
                action_heads['all_actions'] = GMMHead(**agg_arg)
            else:
                raise NotImplementedError
            return action_heads
          
    
    def _build_normalizers(self, embd_args, stats_dict):
        obs_keys = embd_args['tokenizer'].keys()
        action_keys = embd_args['action_head'].keys()
        if not self.agg_modalities:
            normalizers = nn.ModuleDict()
            for key in obs_keys:
                if 'state' in key:
                    stats = stats_dict['obs'][key]
                    normalizers[key] = SimpleNormalizer(torch.from_numpy(stats['mean']).to(torch.float32), torch.from_numpy(stats['std']).to(torch.float32))
                elif 'image' in key:
                    mean = torch.tensor([0.485, 0.456, 0.406]).to(torch.float32)
                    std = torch.tensor([0.229, 0.224, 0.225]).to(torch.float32)
                    normalizers[key] = SimpleNormalizer(mean, std)
            for key in action_keys:
                stats = stats_dict['action'][key]
                normalizers[key] = SimpleNormalizer(torch.from_numpy(stats['mean']).to(torch.float32), torch.from_numpy(stats['std']).to(torch.float32))
            return normalizers
        else:
            normalizers = nn.ModuleDict()
            state_stats = stats_dict['obs']
            agg_state_keys = [key for key in obs_keys if 'state' in key]
            agg_action_keys = [key for key in action_keys]
            agg_state_stats = {'mean': np.concatenate([state_stats[key]['mean'] for key in agg_state_keys], axis=-1),
                                 'std': np.concatenate([state_stats[key]['std'] for key in agg_state_keys], axis=-1)}
            normalizers['all_proprio_states'] = SimpleNormalizer(torch.from_numpy(agg_state_stats['mean']).to(torch.float32), torch.from_numpy(agg_state_stats['std']).to(torch.float32))
            agg_action_stats = {'mean': np.concatenate([stats_dict['action'][key]['mean'] for key in agg_action_keys], axis=-1),
                                'std': np.concatenate([stats_dict['action'][key]['std'] for key in agg_action_keys], axis=-1)}
            normalizers['all_actions'] = SimpleNormalizer(torch.from_numpy(agg_action_stats['mean']).to(torch.float32), torch.from_numpy(agg_action_stats['std']).to(torch.float32))
            image_mean = torch.tensor([0.485, 0.456, 0.406]).to(torch.float32)
            image_std = torch.tensor([0.229, 0.224, 0.225]).to(torch.float32)
            normalizers['all_images'] = SimpleNormalizer(image_mean, image_std)
            return normalizers
            
    def tokenize(self, x, embodiment:str, x_mask):
        """
        Tokenize the input data to shape [batch, num_tokens, hidden_dim] for each modality and concat them
        """
        out = []
        out_mask = []
        bs = x[list(x.keys())[0]].shape[0]
        for modality in self.tokenizers[embodiment].keys():
            if modality not in x.keys():
                dummy_tokens = torch.zeros(bs,
                                           self.tokenizers[embodiment][modality].token_num,
                                           self.tokenizers[embodiment][modality].hidden_dim).to(x[list(x.keys())[0]].device)
                dummy_mask = torch.zeros(bs, self.tokenizers[embodiment][modality].token_num).to(x[list(x.keys())[0]].device)
                out.append(dummy_tokens)
                out_mask.append(dummy_mask)
                continue
            
            if not torch.any(x_mask[modality]):
                input_data = None
                output_tokens, output_mask = self.tokenizers[embodiment][modality].compute_latent(input_data, x_mask[modality])
                out.append(output_tokens)
                out_mask.append(output_mask)
            else:
                if 'state' in modality:
                    input_data = x[modality] * x_mask[modality].unsqueeze(-2) # apply mask
                else:
                    input_data = x[modality]
                
                features = self.tokenizers[embodiment][modality](input_data)
                
                data_shape = features.shape
                data_horizon = data_shape[1]
                horizon = data_horizon
                
                pos_embd = get_sinusoid_encoding_table(horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]).to(features.device)
                pos_embd = einops.repeat(pos_embd, 'one n d -> b one n d', b=data_shape[0])
                features = features + pos_embd.view(data_shape)
                output_tokens, output_mask = self.tokenizers[embodiment][modality].compute_latent(features, x_mask[modality])
                out.append(output_tokens)
                out_mask.append(output_mask)
        out_mask = torch.cat(out_mask, dim=1)
        return out, out_mask
    
    def forward_features(self, x:dict, embodiment:str, x_mask=None):
        """
        Forward the input data through the transformer body
        """
        for modality in x.keys():
            x[modality] = x[modality].to(torch.float32)
        x = self.preprocess_proprio(x, embodiment)
        feature_tokens, feature_mask = self.tokenize(x, embodiment, x_mask)
        feature_tokens = self.preprocess_tokens(feature_tokens, embodiment) # added pos_embd
        query_pos_embd = get_sinusoid_encoding_table(np.sum([len for len in self.query_len.values()]), self.transformer_body.d_model).squeeze().to(feature_tokens.device)
        dummy_features_pos_embd = torch.zeros_like(feature_tokens)[0].to(feature_tokens.device)
        trunk_features = self.transformer_body(feature_tokens, feature_mask, query_pos_embd, dummy_features_pos_embd)[0]
        output_features_dict = self.partition_trunk_tokens(trunk_features, embodiment)
        for modality in output_features_dict.keys():
            output_features_dict[modality] = self.postprocess_tokens(output_features_dict[modality])
        return output_features_dict
    
    def compute_loss(self, batch):
        device = next(self.parameters()).device
        batch = dict_to_device(batch, device)
        self.train_mode = True
        loss_dict = dict()
        loss_dict['all_emb_loss'] = 0
        for emb in batch.keys():
            loss_dict[emb] = dict()
            loss_dict[emb]['loss'] = 0
            obs_emb, obs_mask_emb = batch[emb]['obs'], batch[emb]['obs_mask']
            features_dict_emb = self.forward_features(obs_emb, emb, obs_mask_emb)
            action_emb, action_mask_emb = batch[emb]['action'], batch[emb]['action_mask']
            is_pad_emb = batch[emb]['is_pad']
       
            if emb in self.normalizers.keys():
                for modality in action_emb.keys():
                    action_emb[modality] = self.normalizers[emb][modality].normalize(action_emb[modality])
            
            for modality in [modality for modality in action_emb.keys() if modality in features_dict_emb.keys()]:
                loss_dict[emb][modality] = self.action_heads[emb][modality].compute_loss(features_dict_emb[modality], action_emb[modality], action_mask_emb[modality], is_pad_emb, first_action=None)
                loss_dict[emb]['loss'] += loss_dict[emb][modality]
                
            loss_dict['all_emb_loss'] += loss_dict[emb]['loss']
            
        return loss_dict
    
    def forward(self, obs, embodiment, obs_mask=None, act_mask=None):
        """
        Forward the input data through the transformer body and action head
        TODO: handle mask when inference
        """
        device = next(self.parameters()).device
        obs = dict_to_device(obs, device)
        obs_mask = dict_to_device(obs_mask, device)
        trunk_features_dict = self.forward_features(obs, embodiment, obs_mask)
        actions_dict = dict()
        for modality in self.action_heads[embodiment].keys():
            action = self.action_heads[embodiment][modality](trunk_features_dict[modality])
            if not self.agg_modalities:
                action = action.view(1, self.embodiment_args_dict[embodiment]['action_head'][modality]['action_horizon'], -1)
            else:
                action_head_key = next(iter(self.embodiment_args_dict[embodiment]['action_head']), 'eef_pose')
                action_horzion = self.embodiment_args_dict[embodiment]['action_head'][action_head_key]['action_horizon']
                action = action.view(1, action_horzion, -1)
            actions_dict[modality] = self.normalizers[embodiment][modality].unnormalize(action) if embodiment in self.normalizers.keys() else action
        return actions_dict
            
    def preprocess_tokens(self, features:List[torch.Tensor], embodiment:str):
        """
        Preprocess the tokenized data for the transformer
        """
        tokens = torch.cat(features, axis=-2)
        pos_embs = self.get_pos_embd(tokens)
        tokens = tokens + pos_embs
        return tokens

    def postprocess_tokens(self, tokens):
        """
        Postprocess the tokenized data after the transformer
        """
        if self.token_postprocessing == "mean":
            return tokens.mean(dim=1)
        elif self.token_postprocessing == "cls":
            return tokens[:, 0, :]
        elif self.token_postprocessing == "attentive":
            raise NotImplementedError
        elif self.token_postprocessing == "none":
            return tokens
    
    def preprocess_proprio(self, data:dict, embodiment:str):
        """
        Preprocess the proprioceptive data
        """
        if self.embodiment_args_dict['normalize_state']:
            for modality in data.keys():
                if 'state' in modality:
                    data[modality] = self.normalizers[embodiment][modality].normalize(data[modality])
        
        return data
    
    def get_pos_embd(self, tokens):
        """
        Get the positional embedding for the tokens
        """
        token_len = tokens.shape[1]
        pos_embd = get_sinusoid_encoding_table(token_len, tokens.shape[-1]).to(tokens.device)
        return pos_embd
    
    def partition_trunk_tokens(self, tokens, embodiment:str):
        """
        Partition the tokens for the action head
        """
        if not self.agg_modalities:
            modality_tokens = dict()
            for modality in self.query_len:
                modality_tokens[modality] = tokens[:, :self.query_len[modality], :]
                tokens = tokens[:, self.query_len[modality]:, :]
            return modality_tokens
        else:
            return {'all_actions': tokens}


class DETRVAE_Decoder(nn.Module):
    """ This is the decoder only transformer """
    def __init__(self, backbones, transformer_decoder, state_dim, num_queries, camera_names, action_dim,
                 feature_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.cam_num = len(camera_names)
        self.transformer_decoder = transformer_decoder
        self.state_dim, self.action_dim = state_dim, action_dim
        hidden_dim = transformer_decoder.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.proprio_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
        # encoder extra parameters
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq
        self.additional_pos_embed = nn.Embedding(1, hidden_dim) # learned position embedding for proprio and latent
        self.feature_loss = feature_loss
    
    def forward(self, qpos, image):
        if self.feature_loss:
            # bs,_,_,h,w = image.shape
            image_future = image[:,len(self.camera_names):].clone()
            image = image[:,:len(self.camera_names)].clone()

        all_cam_features = []
        all_cam_pos = []

        if len(self.backbones)>1:
            # Image observation features and position embeddings
            for cam_id, _ in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
                # all_cam_vis.append(vis)
        else:
            if self.feature_loss and self.training:
                all_cam_features_future = []
                bs,_,_,h,w = image.shape
                image_total = torch.cat([image, image_future], axis=0) #cat along the batch dimension
                bs_t,k,c,h_t,w_t = image_total.shape
                features, pos = self.backbones[0](image_total.reshape([-1,c,image_total.shape[-2],image_total.shape[-1]]))
                project_feature = self.input_proj(features[0])
                project_feature = project_feature.reshape([bs_t, self.cam_num, project_feature.shape[1],project_feature.shape[2],project_feature.shape[3]])
                for i in range(self.cam_num):
                    all_cam_features.append(project_feature[:bs,i,:])
                    all_cam_pos.append(pos[0])
                    all_cam_features_future.append(project_feature[bs:,i,:])
                    # all_cam_vis.append(vis)
            else:
                bs,_,c,h,w = image.shape
                features, pos = self.backbones[0](image.reshape([-1,c,h,w]))
                project_feature = self.input_proj(features[0]) 
                project_feature = project_feature.reshape([bs, self.cam_num,project_feature.shape[1],project_feature.shape[2],project_feature.shape[3]])
                for i in range(self.cam_num):
                    all_cam_features.append(project_feature[:,i,:])
                    all_cam_pos.append(pos[0])
                    # all_cam_vis.append(vis)
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos) #B, 512
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3) #B, 512,12,26
        pos = torch.cat(all_cam_pos, axis=3) #B, 512,12,26
        hs = self.transformer_decoder(src, self.query_embed.weight, proprio_input=proprio_input, pos_embed=pos, additional_pos_embed=self.additional_pos_embed.weight) #B, chunk_size, 512
        hs_action = hs[:,-1*self.num_queries:,:].clone() #B, action_dim, 512
        hs_img = hs[:,1:-1*self.num_queries,:].clone() #B, image_feature_dim, 512 #final image feature
        hs_proprio = hs[:,[0],:].clone() #B, proprio_feature_dim, 512
        a_hat = self.action_head(hs_action)
        a_proprio = self.proprio_head(hs_proprio) #proprio head
        if self.feature_loss and self.training:
            # proprioception features
            src_future = torch.cat(all_cam_features_future, axis=3) #B, 512,12,26
            src_future = src_future.flatten(2).permute(2, 0, 1).transpose(1, 0) # B, 12*26, 512
            hs_img = {'hs_img': hs_img, 'src_future': src_future}
        return a_hat, a_proprio, hs_img
    
 
class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent


    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)
        # cvae decoder
        # t = time.time()
        if self.backbones is not None:
            if len(self.backbones)>1:
                # Image observation features and position embeddings
                all_cam_features = []
                all_cam_pos = []
                for cam_id, cam_name in enumerate(self.camera_names):
                    features, pos = self.backbones[cam_id](image[:, cam_id])
                    features = features[0] # take the last layer feature
                    pos = pos[0]
                    all_cam_features.append(self.input_proj(features))
                    all_cam_pos.append(pos)
            else:
                all_cam_features = []
                all_cam_pos = []
                bs,k,c,h,w = image.shape
                image = image[:, :len(self.camera_names)] # exclude future images
                features, pos = self.backbones[0](image.reshape([-1, c, h, w]))
                project_feature = self.input_proj(features[0])
                project_feature = project_feature.reshape([bs, -1,project_feature.shape[1],project_feature.shape[2],project_feature.shape[3]])
                for cam_id in range(project_feature.shape[1]):
                    all_cam_features.append(project_feature[:,cam_id,:])
                    all_cam_pos.append(pos[0])
            # print(f'backbone time: {time.time()-t}')
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=self.action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = args.state_dim # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    if args.same_backbones:
        backbone = build_backbone(args)
        backbones = [backbone]
    else:
        for _ in args.camera_names:
            backbone = build_backbone(args)
            backbones.append(backbone)
        
    if args.no_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)

    if args.model_type=="ACT":
        transformer = build_transformer(args)
        model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
            vq=args.vq,
            vq_class=args.vq_class,
            vq_dim=args.vq_dim,
            action_dim=args.action_dim,
        )
    elif args.model_type=="HIT":
        transformer_decoder = build_transformer_decoder(args)

        model = DETRVAE_Decoder(
            backbones,
            transformer_decoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
            action_dim=args.action_dim,
            feature_loss= args.feature_loss if hasattr(args, 'feature_loss') else False,
        )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model


def build_cross(args):
    agg_modalities = vars(args).get('agg_modalities', False)
    model = CrossTransformer(
        embodiment_args_dict=args.embodiment_args_dict,
        transformer_args=args.transformer_args,
        norm_stats=args.norm_stats,
        agg_modalities=agg_modalities,
    )
    
    return model