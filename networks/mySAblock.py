
import math
import torch
import torch.nn as nn
import numpy as np
from monai.utils import optional_import
import time
einops, _ = optional_import("einops")


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        stdvar: float = 1.0,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super(SABlock, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.stdvar = stdvar

        #-----------------------------------------------------------
        self.fc_mu_var = nn.Linear(hidden_size//self.num_heads * 2, 2) 
        
        #-----------------------------------------------------------

        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        

    def forward(self, x):
        #x: [4, 216, 768]
        q, k, v = einops.rearrange(self.qkv(x), "b h (qkv l d) -> qkv b l h d", qkv=3, l=self.num_heads) #q k v [4, 12, 216, 64]
        #print('q:',q.shape,'k:',k.shape,'v:',v.shape)
        q_mlp = torch.unsqueeze(q,3) #[4, 12, 216, 1, 64]
        #start_time = time.time()
        q_mlp = q_mlp.repeat(1, 1, 1, 216, 1)
        #for i in range(q_mlp.shape[2]-1):
        #    q_mlp = torch.cat([q_mlp,torch.unsqueeze(q,3)], dim = 3)  #[4, 12, 216, 216, 64]
        #print('q',q_mlp.shape)
        
        k_mlp = torch.unsqueeze(k,2) #[4, 12, 1, 216, 64]
        k_mlp = k_mlp.repeat(1, 1, 216, 1, 1)
        #for i in range(k_mlp.shape[3]-1):
        #    k_mlp = torch.cat([k_mlp,torch.unsqueeze(k,2)], dim = 2)  #[4, 12, 216, 216, 64]
        #print('k',k_mlp.shape)
        #print('loop time:', time.time() -start_time)

        q_k_mlp = torch.cat([q_mlp,k_mlp],dim = 4) #[4, 12, 216, 216, 128]
        att_map_params = self.fc_mu_var(q_k_mlp) #[4, 12, 216, 216, 2]
        fc_mu_map = att_map_params[..., 0] #[4, 12, 216, 216]
        fc_logvar_map = att_map_params[..., 1] #[4, 12, 216, 216]
        att_mat = self.reparameterize(fc_mu_map, fc_logvar_map) #[4, 12, 216, 216]
        #att_mat = att_mat/torch.sum(att_mat,dim = 3,keepdim=True)  
        att_mat = att_mat.softmax(dim=-1) #softmax
        #att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1) 
        if torch.isnan(att_mat).any():
            print('attention mat nan!!!')
        att_mat = self.drop_weights(att_mat) #att_mat: [4, 12, 216, 216]

        nll = compute_kld(fc_logvar_map, fc_mu_map, q, k, self.scale,self.stdvar)

        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        #print('x2:',x.shape)
        x = einops.rearrange(x, "b h l d -> b l (h d)")
        #print('x3:',x.shape)
        x = self.out_proj(x)
        x = self.drop_output(x)
        #print('x4:',x.shape)
        return x, nll



def compute_kld (logvar, att_map, q, k, scale, stdvar):
    stdvar = torch.tensor(stdvar).cuda(device=att_map.device)
    var_map = torch.exp(logvar)
    #print(torch.isinf(var_map).any())
    deterministic_atten_map = (torch.einsum("blxd,blyd->blxy", q, k) * scale)
    #kld = -0.5 * (1 + logvar - (att_map - deterministic_atten_map).pow(2) - var_map)
    kld = -0.5 * (1 + logvar -torch.log(stdvar**2)- ((att_map - deterministic_atten_map).pow(2) + var_map)/(stdvar**2))

    kld = torch.mean(kld, dim = [1,2,3])

    return kld

