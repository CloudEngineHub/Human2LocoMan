from typing import List
from torchvision import transforms
from algos.detr.models.transformer import CrossAttention
import torch, torch.nn as nn, torchvision

class BaseTokenizer(nn.Module):
    """
    Base tokenizer for the cross-transformer. Adapted from HPT codebase.
    """
    def __init__(self):
        super().__init__()
        
    def init_cross_attn(self, embodiment_tokenizer_args, modality:str):
        self.token_num = embodiment_tokenizer_args[modality]['token_num']
        self.tokens = nn.Parameter(torch.randn(1, self.token_num, embodiment_tokenizer_args['hidden_dim']))
        self.hidden_dim = embodiment_tokenizer_args['hidden_dim']
        self.cross_attn = CrossAttention(embodiment_tokenizer_args['hidden_dim'],
                                         embodiment_tokenizer_args['nhead'],
                                         embodiment_tokenizer_args['dim_head'],
                                         embodiment_tokenizer_args['dropout'])
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
            
    def save(self, path):
        torch.save(self.state_dict(), path)
            
    @property
    def device(self):
        return next(self.parameters()).device
    
    def compute_latent(self, feat):
        # Initial reshape to adapt to token dimensions
        # (32, 3, 1, 49, 128)
        # features = self.forward(x)  
        features = feat.reshape(features.shape[0], -1, features.shape[-1])  # (32, 147, 128)
        # Replicating tokens for each item in the batch and computing cross-attention
        query_tokens = self.tokens.repeat(len(features), 1, 1)  # (32, 16, 128)
        output_tokens = self.cross_attn(query_tokens, features)  # (32, 16, 128)
        return output_tokens
        
class ImageTokenizer(BaseTokenizer):
    def __init__(
        self,
        output_dim: int = 10,
        weights: str = "DEFAULT",
        resnet_model: str = "resnet18",
        num_of_copy: int = 1,
        **kwargs
    ):
        super().__init__()
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)
        
        self.num_of_copy = num_of_copy
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])
        
        if num_of_copy > 1:
            self.net = nn.ModuleList(
                [nn.Sequential(*list(pretrained_model.children())[:-2]) for _ in range(num_of_copy)]
            )
            
        self.out_dim = output_dim
        self.to_tensor = transforms.ToTensor()
        self.proj = nn.Linear(512, output_dim)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
    def forward(self, x):
        """
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size, 
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]     
        """
        B, *_, H, W = x.shape
        x = x.view(len(x), -1, 3, H, W)
        if self.num_of_copy > 1:
            # separate encoding for each view
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            feat = torch.stack(out, dim=1)
        else:
            x = x.view(-1, 3, H, W)
            feat = self.net(x) # bs, 512, feat_height, feat_width, in our case, (bs, 512, 15, 40)
        feat = feat.flatten(start_dim=-2).transpose(1, 2) # bs, 600, 512
        # concat along time
        feat = self.proj(feat) # bs, seq, out_dim
        feat = feat.reshape(B, -1, feat.shape[-1]).contiguous()
        return feat
    
    def compute_latent(self, feat, mask):
        # TODO: do we need to handle the case where mask is not the same for each sample in the batch?
        # for images, mask should be a single boolean (bs,)
        # Initial reshape to adapt to token dimensions
        # (32, 3, 1, 49, 128)
        # features = self.forward(x)
        bs = mask.shape[0]
        if feat is None:
            output_tokens = torch.zeros((bs, self.token_num, self.hidden_dim)).to(self.device)
            trunk_mask = torch.zeros((bs, self.token_num)).to(self.device)
        else:
            features = feat.reshape(bs, -1, feat.shape[-1])  # (32, 147, 128)
            # Replicating tokens for each item in the batch and computing cross-attention
            query_tokens = self.tokens.repeat(bs, 1, 1)  # (32, 16, 128)
            output_tokens = self.cross_attn(query_tokens, features)  # (32, 16, 128)
            trunk_mask = torch.ones((feat.shape[0], self.token_num)).to(self.device)
        return output_tokens, trunk_mask

class DINOImageTokenizer(BaseTokenizer):
    def __init__(
        self,
        output_dim: int = 10,
        weights: str = "DEFAULT",
        # resnet_model: str = "resnet18",
        num_of_copy: int = 1,
        **kwargs
    ):
        super().__init__()
        # pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)
        
        self.num_of_copy = num_of_copy
        # self.net = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.net = net = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        if num_of_copy > 1:
            self.net = nn.ModuleList(
                [net for _ in range(num_of_copy)]
            )
            
        self.out_dim = output_dim
        self.to_tensor = transforms.ToTensor()
        self.proj = nn.Linear(384, output_dim)
        self.avgpool = nn.AvgPool2d(7, stride=2)


    def forward(self, x):
        """
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size, 
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]     
        """
        B, *_, H, W = x.shape
        new_h = H // 14 * 14
        new_w = W // 14 * 14
        x = x.view(len(x), -1, 3, H, W)
        if self.num_of_copy > 1:
            # separate encoding for each view
            x = x[:, :, :, :new_h, :new_w]
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net.forward_features(input)["x_norm_patchtokens"])
            feat = torch.stack(out, dim=1)
        else:
            x = x.view(-1, 3, H, W)
            x = x[:, :, :new_h, :new_w]
            feat = self.net.forward_features(x)["x_norm_patchtokens"] # bs, num_tokens, in our case, (bs, 3094, 384)
        feat = feat.transpose(1,2).reshape(B, -1, new_h // 14, new_w // 14).contiguous() # bs, 384, 15, 40 or so
        feat = self.avgpool(feat).flatten(start_dim=-2).transpose(1, 2) # bs, num_tokens, 384
        # feat = feat.transpose(1, 2) # bs, 600, 384
        # concat along time
        feat = self.proj(feat) # bs, seq, out_dim
        feat = feat.reshape(B, -1, feat.shape[-1]).contiguous()
        return feat
    
    def compute_latent(self, feat, mask):
        # TODO: do we need to handle the case where mask is not the same for each sample in the batch?
        # for images, mask should be a single boolean (bs,)
        # Initial reshape to adapt to token dimensions
        # (32, 3, 1, 49, 128)
        # features = self.forward(x)
        bs = mask.shape[0]
        if feat is None:
            output_tokens = torch.zeros((bs, self.token_num, self.hidden_dim)).to(self.device)
            trunk_mask = torch.zeros((bs, self.token_num)).to(self.device)
        else:
            features = feat.reshape(bs, -1, feat.shape[-1])  # (32, 147, 128)
            # Replicating tokens for each item in the batch and computing cross-attention
            query_tokens = self.tokens.repeat(bs, 1, 1)  # (32, 16, 128)
            output_tokens = self.cross_attn(query_tokens, features)  # (32, 16, 128)
            trunk_mask = torch.ones((feat.shape[0], self.token_num)).to(self.device)
        return output_tokens, trunk_mask    

class ProprioTokenizer(BaseTokenizer):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
        **kwargs
    ):
        super().__init__()
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList([nn.Sequential(*modules) for _ in range(num_of_copy)])
        
    def forward(self, x):
        if self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            y = torch.stack(out, dim=1)
        else:
            y = self.net(x)
        return y
    
    def compute_latent(self, feat, mask):
        # Initial reshape to adapt to token dimensions
        # (32, 3, 1, 49, 128)
        # features = self.forward(x)
        bs = mask.shape[0]
        if feat is None:
            output_tokens = torch.zeros((bs, self.token_num, self.hidden_dim)).to(self.device)
            trunk_mask = torch.zeros((bs, self.token_num)).to(self.device)
        else:
            features = feat.reshape(bs, -1, feat.shape[-1])  # (32, 147, 128)
            # Replicating tokens for each item in the batch and computing cross-attention
            query_tokens = self.tokens.repeat(bs, 1, 1)  # (32, 16, 128)
            output_tokens = self.cross_attn(query_tokens, features)  # (32, 16, 128)
            trunk_mask = torch.ones((feat.shape[0], self.token_num)).to(self.device)
        return output_tokens, trunk_mask