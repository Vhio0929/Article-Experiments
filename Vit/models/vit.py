# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.scale_dot_product_attention import ScaleDotProductAttention
import math
import numpy as np



# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        print("====现在是原多头注意力机制====")

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

#=============================================================
class Attention1(nn.Module):
    def __init__(self, d_model, heads, dim_head = 64, dropout = 0.):
        #super(Attention1, self).__init__()
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.d_model = d_model
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == d_model)


        self.attention = ScaleDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, d_model),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        print("====现在后修改的Attention====")

    def forward(self, x, mask=None):
        #print("===x.shape:", x.shape)
        #x, single = y
        n, q_len = x.shape[0:2]
        n, k_len = x.shape[0:2]

        # 1. dot product with weight matrices
        q0, k0, v0 = self.w_q(x), self.w_k(x), self.w_v(x)

        q_ = q0.reshape(n, q_len, self.heads, self.dim_head).transpose(1, 2)
        k_ = k0.reshape(n, k_len, self.heads, self.dim_head).transpose(1, 2)
        v_ = v0.reshape(n, k_len, self.heads, self.dim_head).transpose(1, 2)

        attention_res, attn = self.attention(q_, k_, v_, mask=mask)
        
        
        concat_res = attention_res.transpose(1, 2).contiguous().view(n, q_len, -1)

        
        return self.to_out(concat_res)


#=============================================================

#=============================================================
class LS_Attention(nn.Module):
    def __init__(self, d_model, n_head, dim_head = 64, dropout = 0.):
        super(LS_Attention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()

        #=========================================================================
        self.hd_list = [(2**(1-math.ceil(4*(i+1)/n_head))*(d_model/(2**(math.log2(n_head)-1)))) for i in range(n_head-1)]
        print("**********************:", self.hd_list)
        self.hd_list.append((n_head*d_model+4*d_model)/(16*n_head))
        print("**********************:", self.hd_list)
        self.hd_list = np.cumsum(self.hd_list)
        self.hd_list = np.insert(self.hd_list,0,0)
        print("**********************:", self.hd_list)
        assert self.hd_list[-1] == d_model
        #========================================================================

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        print("====现在是LS_HAM====")

    def forward(self, x, mask=None):

        n, q_len = x.shape[0:2]
        n, k_len = x.shape[0:2]

        # 1. dot product with weight matrices
        q0, k0, v0 = self.w_q(x), self.w_k(x), self.w_v(x)
        
        attention_list = []
        for j in range(self.n_head):
            q1 = q0[:, :, int(self.hd_list[j]):int(self.hd_list[j + 1])].reshape(n, q_len, 1, -1).transpose(1, 2)
            k1 = k0[:, :, int(self.hd_list[j]):int(self.hd_list[j + 1])].reshape(n, k_len, 1, -1).transpose(1, 2)
            v1 = v0[:, :, int(self.hd_list[j]):int(self.hd_list[j + 1])].reshape(n, k_len, 1, -1).transpose(1, 2)
            attention_res, attn = self.attention(q1, k1, v1, mask=mask)
            concat_res = attention_res.transpose(1, 2).contiguous().view(n, q_len, -1)
            
            
            attention_list.append(concat_res)
            

        concat_res = torch.cat(attention_list, -1)
        output = self.w_concat(concat_res)

        return output
#=============================================================

layer_num = 0
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention1(dim, heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        
        for attn, ff in self.layers:
            
            x = attn(x) + x
            x = ff(x) + x
        return x

#================================================================
class LS_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LS_Attention(dim, heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
#================================================================

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

#==================================================================
class LS_ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 128, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = LS_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

#==================================================================
