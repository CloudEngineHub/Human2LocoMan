import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
from algos.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer, build_MXT_model_and_optimizer
import IPython
e = IPython.embed

from collections import OrderedDict

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'HIT':
        policy = HITPolicy(policy_config)
    elif policy_class == 'MXT':
        policy = MXTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'HIT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'MXT':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


class HITPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        args_override['model_type'] = "HIT"
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.feature_loss_weight = args_override['feature_loss_weight'] if 'feature_loss_weight' in args_override else 0.0
        try:
            self.state_idx = args_override['state_idx']
            self.action_idx = args_override['action_idx']
        except:
            self.state_idx = None
            self.action_idx = None
        self.precomputed_features = args_override['precomputed_features']
    
    def __call__(self, qpos, image, actions=None, is_pad=None, action_masks=None):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        if self.action_idx is not None and actions is not None:
            actions = actions[:, :, self.action_idx]
        
        if not self.precomputed_features:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            image = normalize(image)
        if actions is not None: # training/validation time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            a_hat, _, hs_img_dict = self.model(qpos, image)
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            all_l1 = all_l1 * action_masks.unsqueeze(1)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).sum() / (~is_pad).sum()
            loss_dict['l1'] = l1
            if self.model.feature_loss and self.model.training:
                loss_dict['feature_loss'] = F.mse_loss(hs_img_dict['hs_img'], hs_img_dict['src_future']).mean()
                loss_dict['loss'] = loss_dict['l1'] + self.feature_loss_weight*loss_dict['feature_loss']
            else:
                loss_dict['loss'] = loss_dict['l1']
            return loss_dict
        else: # inference time
            a_hat, _, _ = self.model(qpos, image) # no action, sample from prior
            return a_hat

    def forward_inf(self, qpos, image):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        a_hat, _, _ = self.model(qpos, image) # no action, sample from prior
        return a_hat
        
    def configure_optimizers(self):
        return self.optimizer   
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)
        
            

class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        self.camera_names = args_override['camera_names']

        self.observation_horizon = args_override['observation_horizon'] ### TODO TODO TODO DO THIS
        self.action_horizon = args_override['action_horizon'] # apply chunk size
        self.prediction_horizon = args_override['prediction_horizon'] # chunk size
        self.num_inference_timesteps = args_override['num_inference_timesteps']
        self.ema_power = args_override['ema_power']
        self.lr = args_override['lr']
        self.weight_decay = 0

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override['action_dim'] # 14 + 2
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 14 # camera features and proprio

        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
            pools.append(SpatialSoftmax(**{'input_shape': [512, 15, 20], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)
        
        backbones = replace_bn_with_gn(backbones) # TODO


        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim*self.observation_horizon
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        if actions is not None: # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=obs_cond.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss

            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict
        else: # inference time
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            
            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model
            
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action
            
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        args_override['model_type'] = "ACT"
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.vq = args_override['vq']
        print(f'KL Weight {self.kl_weight}')
        try:
            self.state_idx = args_override['state_idx']
            self.action_idx = args_override['action_idx']
        except:
            self.state_idx = None
            self.action_idx = None
        self.precomputed_features = args_override['precomputed_features']
    

    def __call__(self, qpos, image, actions=None, is_pad=None, action_masks=None, vq_sample=None):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        if self.action_idx is not None and actions is not None:
            actions = actions[:, :, self.action_idx]
            
        env_state = None
        if not self.precomputed_features:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, env_state, actions, is_pad, vq_sample)
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            all_l1 = all_l1 * action_masks.unsqueeze(1)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).sum() / (~is_pad).sum()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample) # no action, sample from prior
            return a_hat

    def forward_inf(self, qpos, image):
        if self.state_idx is not None:
            qpos = qpos[:, self.state_idx]
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=None) # no action, sample from prior
        return a_hat
        
    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)

        return binaries
        
    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)

class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
    
    
class MXTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        args_override['model_type'] = "mxt"
        model, optimizer = build_MXT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.share_cross_attn = args_override['share_cross_attn']
        
    def forward(self, data, embodiment, obs_mask=None, act_mask=None):
        return self.model(data, embodiment, obs_mask, act_mask)
    
    def compute_loss(self, batch):
        return self.model.compute_loss(batch)
    
    def configure_optimizers(self):
        return self.optimizer
    
    def serialize(self):
        tokenizers = self.model.tokenizers
        tokenizer_keys = list(tokenizers.keys())
        tokenizer_state_dicts = {key: tokenizers[key].state_dict() for key in tokenizer_keys}
        trunk_state_dict = self.model.transformer_body.state_dict()
        action_heads = self.model.action_heads
        action_head_keys = list(action_heads.keys())
        action_head_state_dicts = {key: action_heads[key].state_dict() for key in action_head_keys}
        return {
            'tokenizers': tokenizer_state_dicts,
            'trunk': trunk_state_dict,
            'action_heads': action_head_state_dicts
        }
    
    def deserialize(self, model_dicts, eval=False):
        # load tokenizers
        if eval:
            tokenizers = self.model.tokenizers
            tokenizer_keys = list(tokenizers.keys())
            for key in tokenizer_keys:
                self.model.tokenizers[key].load_state_dict(model_dicts['tokenizers'][key])
        elif self.share_cross_attn:
            tokenizers = self.model.tokenizers
            tokenizer_keys = list(tokenizers.keys())
            pretrained_embodiment = list(model_dicts['tokenizers'].keys())[0]
            print("Copying pretrained weights from", pretrained_embodiment, "to the following items:")
            pretrained_tokenizers = model_dicts['tokenizers'][pretrained_embodiment]
            for name, param in pretrained_tokenizers.items():
                if 'cross_attn' in name:
                    print(name)
                    for emb_to_train in tokenizer_keys:
                        tokenizers[emb_to_train].state_dict()[name].copy_(param.data)
                    
        # load trunk
        self.model.transformer_body.load_state_dict(model_dicts['trunk'])
        # load action head
        if eval:
            action_heads = self.model.action_heads
            action_head_keys = list(action_heads.keys())
            for key in action_head_keys:
                self.model.action_heads[key].load_state_dict(model_dicts['action_heads'][key])