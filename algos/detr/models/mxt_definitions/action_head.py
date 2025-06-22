"""
Action head to adapt the output of the transformer to the action space. Adapted from HPT codebase.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

from functools import partial
from typing import List
from algos.detr.models.transformer import CrossAttention
import torch.distributions as dist

LOSS = partial(F.smooth_l1_loss, beta=0.05, reduction='none')
INIT_CONST = 0.02


class ActionHead(nn.Module):
    """ Abstract class for policy head."""

    def __init__(self, **kwargs):
        super().__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path : str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_loss(self, x: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, is_pad: torch.Tensor, first_action: torch.Tensor=None) -> torch.Tensor:
        self.target_action = label
        self.pred_action = self(x, first_action).view(self.target_action.shape)
        loss_list = LOSS(self.pred_action, self.target_action)
        action_horizon = self.target_action.shape[1]
        loss_list = loss_list * mask.unsqueeze(1)
        is_pad = is_pad[:, :action_horizon].unsqueeze(-1) if is_pad is not None else torch.zeros_like(self.target_action)
        mean_loss = torch.sum(loss_list * ~is_pad) / torch.sum(~is_pad)
        return mean_loss



class MLP(ActionHead):
    """Simple MLP based policy head"""

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        dropout: bool = False,
        tanh_end: bool = False,
        ln: bool = True,
        **kwargs,
    ) -> None:
        """vanilla MLP head on the pooled feature"""
        super().__init__()
        # self.input = input
        self.input_dim
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if dropout:
                modules.append(nn.Dropout(p=0.1))
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, first_action=None):
        """
        Forward pass of the policy head module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        y = self.net(x)
        if first_action is not None:
            y = y + first_action
        return y

class Conv1D(ActionHead):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        kernel_size: int = 3,
        n_groups: int = 8,
        action_horizon: int = 4,
        **kwargs,
    ) -> None:
        """
        Vanilla FFW head. Combined with learnable tokens imposed before the trunk.
        This one has a lower loss than MLP and should work better for long action chunks.
        """
        super().__init__()
        self.block_in = nn.Sequential(
            nn.Conv1d(input_dim, widths[0], kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, widths[0]),
            nn.Mish(),
        )
        output_dim = output_dim // action_horizon
        self.block_out = nn.Sequential(
            nn.Conv1d(widths[0], output_dim, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x, first_action=None):
        """
        Forward pass of the policy head module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        y = self.block_out(self.block_in(x)).view(len(x), -1) # flatten it
        if first_action is not None:
            y = y + first_action
        return y

class TransformerDecoderHead(ActionHead):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        crossattn_modality_dropout: float = 0.1,
        crossattn_heads: int = 8,
        crossattn_dim_head: int = 64,
        action_horizon: int = 4,
        **kwargs,
    ) -> None:
        """
        Transformer decoder similar to ACT or Detr head.
        This version uses cross attention and does not require retraining the trunk.
        """
        super().__init__()
        token_num = action_horizon
        self.tokens = nn.Parameter(
            torch.randn(1, token_num, output_dim) * INIT_CONST
        )

        self.cross_attention = CrossAttention(
            output_dim,
            heads=crossattn_heads,
            dim_head=crossattn_dim_head,
            dropout=crossattn_modality_dropout,
        )
        embed_dim = crossattn_dim_head * crossattn_heads
        self.mlp = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, output_dim))

    def forward(self, x: torch.Tensor, first_action=None) -> torch.Tensor:
        """
        Args:
        x: (B, token_len, input_dim)
        """
        context = self.mlp(x)
        context = context.reshape(context.shape[0], -1, context.shape[-1])
        # Replicating tokens for each item in the batch and computing cross-attention
        queries = self.tokens.repeat(len(context), 1, 1)
        out = self.cross_attention(queries, context) #.view(len(x), -1)
        # out[:, 1:, :] += out[:, 0, :].unsqueeze(1)
        # print("----RAW_OUT----", out[0])
        if first_action is not None:
            out = out + first_action.unsqueeze(1)
        # print("----OUT----", out[0])
        return out

############  Gaussian Mixture Model Head ############
# https://github.com/siddhanthaldar/BAKU/blob/main/baku/agent/networks/policy_head.py
class GMMHead(ActionHead):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        min_std: float = 0.0001,
        num_modes: int = 5,
        **kwargs,
        ):
        super().__init__()
        self.num_modes = num_modes
        self.output_dim = output_dim
        self.min_std = min_std

        if num_layers > 0:
            sizes = [input_dim] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        self.mean_layer = nn.Linear(hidden_size, output_dim * num_modes)
        self.logstd_layer = nn.Linear(hidden_size, output_dim * num_modes)
        self.logits_layer = nn.Linear(hidden_size, num_modes)
        self.actv = F.softplus

    def forward_fn(self, x: torch.Tensor, **kwargs):
        """
        Args:
        x: (B, input_size)
        """
        share = self.share(x)
        means = self.mean_layer(share).view(-1, self.num_modes, self.output_dim)
        means = torch.tanh(means)
        logits = self.logits_layer(share)

        if self.training:
            logstds = self.logstd_layer(share).view(
                -1, self.num_modes, self.output_dim
            )
            stds = self.actv(logstds) + self.min_std
        else:
            stds = torch.ones_like(means) * self.min_std
        return means, stds, logits

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Compute the GMM distribution or the output means.
        """
        means, scales, logits = self.forward_fn(x)
        compo = dist.Normal(loc=means, scale=scales)
        compo = dist.Independent(compo, 1)
        mix = dist.Categorical(logits=logits)
        gmm = dist.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=compo
        )
        if self.training:
            return gmm
        else:
            return gmm.mean

    def compute_loss(self, x: torch.Tensor, data: dict) -> torch.Tensor:
        gmm = self(x)
        log_probs = gmm.log_prob(data['action'])
        loss = -log_probs
        return loss.mean()