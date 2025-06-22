import torch
import torch.nn as nn
#Reference https://github.com/karpathy/nanoGPT

class Transformer_Block(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_rate, self_attention=True, query_num=50) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
        self.ln_3 = nn.LayerNorm(latent_dim)
        self.ln_4 = nn.LayerNorm(latent_dim)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.query_num = query_num
        self.self_attention = self_attention
        
    def forward(self, input):
        x = input['x']
        memory = input['memory']
        memory_embed = input['memory_embed']
        if self.self_attention:
            x = self.ln_1(x)
            x2 = self.self_attn(x, x, x, need_weights=False)[0]
            x = x + self.dropout1(x2)
            x = self.ln_2(x)
            if memory is not None:
                x3 = self.cross_attn(x, memory + memory_embed if memory_embed is not None else memory, memory, need_weights=False)[0]
                x = x + self.dropout2(x3)
                x = self.ln_3(x)
            x = x + self.mlp(x)
            x = self.ln_4(x)
            return {
                'x': x,
                'memory': memory,
                'memory_embed': memory_embed
            }
        else:
            print("self_attention is False, ignoring memory")
            x = input['x']
            memory = input['memory']
            memory_embed = input['memory_embed']
            x = self.ln_1(x)
            x_action = x[-self.query_num:].clone()
            x_condition = x[:-self.query_num].clone()
            x2 = self.self_attn(x_action, x_condition, x_condition, need_weights=False)[0] # q, k, v
            x2 = x2 + self.dropout1(x2)
            x2 = self.ln_2(x2)
            x2 = x2 + self.mlp(x2)
            x2 = self.ln_4(x2)
            x = torch.cat((x_condition, x2), dim=0)
            return {
                'x': x,
                'memory': memory,
                'memory_embed': memory_embed
            }
            
    
class Transformer_BERT_memory(nn.Module):
    def __init__(self, context_len, latent_dim=128, num_head=4, num_layer=4, dropout_rate=0.0,  
                 use_pos_embd_image=False, use_pos_embd_action=False, query_num=50,
                 self_attention=True) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.context_len = context_len
        self.use_pos_embd_image = use_pos_embd_image==1
        self.use_pos_embd_action = use_pos_embd_action==1
        self.query_num = query_num
        if use_pos_embd_action and use_pos_embd_image:
            self.weight_pos_embed = None
        elif use_pos_embd_image and not use_pos_embd_action:
            self.weight_pos_embed = nn.Embedding(self.query_num, latent_dim)
        elif not use_pos_embd_image and not use_pos_embd_action:
            self.weight_pos_embed = nn.Embedding(self.context_len, latent_dim)
        elif not use_pos_embd_image and use_pos_embd_action:
            raise ValueError("use_pos_embd_action is not supported")
        else:
            raise ValueError("bug ? is not supported")
        
        self.attention_blocks = nn.Sequential(
            *[Transformer_Block(latent_dim, num_head, dropout_rate, self_attention, query_num) for _ in range(num_layer)],
        )
        self.self_attention = self_attention
    
    def forward(self, x, memory=None, pos_embd_image=None, query_embed=None, memory_embed=None):
        assert memory.shape[-1] == self.latent_dim and memory_embed.shape[-1] == self.latent_dim, "memory shape is not compatible with latent_dim"
        if not self.use_pos_embd_image and not self.use_pos_embd_action: #everything learned - severe overfitting
            x = x + self.weight_pos_embed.weight[:, None]
        elif self.use_pos_embd_image and not self.use_pos_embd_action: #use learned positional embedding for action 
            x[-self.query_num:] = x[-self.query_num:] + self.weight_pos_embed.weight[:, None]
            x[:-self.query_num] = x[:-self.query_num] + pos_embd_image
        elif self.use_pos_embd_action and self.use_pos_embd_image: #all use sinusoidal positional embedding
            x[-self.query_num:] = x[-self.query_num:] + query_embed
            x[:-self.query_num] = x[:-self.query_num] + pos_embd_image 
        
        input = {
            'x': x,
            'memory': memory,
            'memory_embed': memory_embed
        }     
        output = self.attention_blocks(input)
        # take the last token
        return output['x']
    
    
class Memory_Decoder(nn.Module):
    def __init__(self, context_len=None, d_model=512, nhead=8, num_decoder_layers=6, dropout=0.1,
                 use_pos_embd_image=False, query_num=50, use_pos_embd_action=False,
                 self_attention=True):
        super().__init__()

        self.decoder = Transformer_BERT_memory(context_len=context_len,
                                        latent_dim=d_model,
                                        num_head=nhead,
                                        num_layer=num_decoder_layers,
                                        dropout_rate=dropout,
                                        use_pos_embd_image=use_pos_embd_image,
                                        use_pos_embd_action=use_pos_embd_action,
                                        query_num=query_num,
                                        self_attention=self_attention)
    
        self._reset_parameters()
    
        self.d_model = d_model
        self.nhead = nhead
        self.self_attention = self_attention

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, memory=None, proprio_input=None, additional_pos_embed=None, pos_embed=None, memory_embed=None):
        # TODO flatten only when input has H and W
        if len(src.shape) == 4: # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            src = torch.cat([proprio_input[None], src], axis=0)
            
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1) # seq, bs, dim
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

        #decoder will add positional encoding 

        action_input_token = torch.zeros_like(query_embed)
        input_tokens = torch.cat([src, action_input_token], axis=0)  #input tokens + positional encoding for action 
        hs = self.decoder(input_tokens, memory, pos_embed, query_embed, memory_embed)
        hs = hs.transpose(1, 0)
        return hs