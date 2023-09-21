from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from datetime import datetime

from torchmetrics.functional import pairwise_cosine_similarity

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from einops import rearrange
from scipy.optimize import linear_sum_assignment

class BipartiteGraphMatcher(nn.Module):
    """bipartite graph matching solver via sinkhorn."""
    def __init__(self, sinkhorn=0):
        super().__init__()
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.sinkhorn = sinkhorn
        

    def log_sinkhorn_iterations(self, Z, log_mu, log_nu, iters: int):
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)


    def log_optimal_transport(self, scores, alpha, iters: int):
        """ Perform Differentiable Optimal Transport in Log-space for stability"""
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m*one).to(scores), (n*one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities by M+N
        return Z
    
    
    def forward(self, cost_matrix):
        scores = self.log_optimal_transport(
            cost_matrix, self.bin_score,
            iters=self.sinkhorn)
        return scores


class TrajGenerator(nn.Module):
    """Trajectory Generator in CATS (i.e., CatGen)."""
    def __init__(self, num_traj=128, dim_loc=2, dim_embed=32, num_head=8, sinkhorn=0, use_sinkhorn=False):
        super(TrajGenerator, self).__init__()
        
        self.num_traj = num_traj
        self.dim_loc = dim_loc
        self.dim_embed = dim_embed
        self.num_head = num_head
        self.sinkhorn = sinkhorn
        self.use_sinkhorn = use_sinkhorn
        
        # location embedding (2 -> 32)
        self.location_embedding = nn.Sequential(
            nn.Linear(self.dim_loc, self.dim_embed),
            nn.ReLU()
            )
        
        # time embedding (24 -> 32)
        self.time_embedding = nn.Sequential(
            nn.Embedding(24, self.dim_embed),
            nn.ReLU()
            )
        
        self.batch_norm = nn.BatchNorm1d(self.dim_embed * 2)
        
        # global spatial context attention
        self.global_sc_attn = nn.MultiheadAttention(self.dim_embed * 2, self.num_head)
        self.layer_norm = nn.LayerNorm(self.dim_embed * 2)
        
        # trajectory modeling
        self.traj_mod_gru = nn.GRUCell(self.dim_embed * 2, self.dim_embed * 2)
        self.tanh = nn.Tanh()
        
        # bipartite graph matching solver via sinkhorn
        self.bipartite_graph_matcher = BipartiteGraphMatcher(sinkhorn=self.sinkhorn)

    def forward(self, x):
        # location embedding
        embed_x = self.location_embedding(x).float()
        
        # time embedding
        embed_t = self.time_embedding(torch.as_tensor(0).to('cuda')) * torch.ones((self.num_traj,1)).to('cuda')
        
        # spatio-temporal concatenation
        embed_p = torch.cat([embed_x[0], embed_t], dim = 1)
        embed_p = self.batch_norm(embed_p)

        # global spatial context attention for current-step embeddings
        curr_attn_output, _ = self.global_sc_attn(embed_p, embed_p, embed_p)
#         curr_attn_output = self.drop_out(curr_attn_output)
        curr_state = self.layer_norm(embed_p + curr_attn_output)
        # normalize the value range to (-1, 1)
        curr_state = self.tanh(curr_state)
        
        # the prediction of next state, which also serves as the hidden state in GRU
        next_state_pred = torch.zeros((self.num_traj, self.dim_embed * 2)).to('cuda')

        # accumulated matching cost
        matching_cost = torch.zeros((self.num_traj,)).to('cuda')
        
        # the generated trajectories
        gen_traj_list = [x[0]]
        
        # go through the time
        for i_next in range(1, x.shape[0]):
            next_embed_p = embed_x[i_next]
            
            # time embedding
            next_embed_t = self.time_embedding(torch.as_tensor(i_next).to('cuda')) * torch.ones((self.num_traj,1)).to('cuda')
            
            # spatio-temporal concatenation
            next_embed_p = torch.cat([next_embed_p, next_embed_t], dim = 1)
            next_embed_p = self.batch_norm(next_embed_p)
            
            # global spatial context attention for next-step embeddings
            next_attn_output, _ = self.global_sc_attn(next_embed_p, next_embed_p, next_embed_p)
            # next_attn_output = self.drop_out(next_attn_output)
            next_state = self.layer_norm(next_embed_p + next_attn_output)
            # normalize the value range to (-1, 1)
            next_state = self.tanh(next_state)
            
            # the prediction of next-step trajectory points. value in (-1, 1)
            next_state_pred = self.traj_mod_gru(curr_state, next_state_pred)
            
            # cost matrix based on L2 cost
            cost_matrix = torch.cdist(next_state_pred, next_state, p=2)
            
            if self.use_sinkhorn:
                # bipartite graph matching via sinkhorn. not preferred due to instability
                sim_matrix = pairwise_cosine_similarity(next_state_pred, next_state)
                # bipartite graph matching
                matching_res = self.bipartite_graph_matcher(sim_matrix.unsqueeze(0).log_softmax(dim=-2))

                # Get valid matches
                max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
                indices0, indices1 = max0.indices, max1.indices
                mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
                mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
                zero = scores.new_tensor(0)
                mscores0 = torch.where(mutual0, max0.values.exp(), zero)
                mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
                valid0 = mutual0 & (mscores0 > 0.)
                valid1 = mutual1 & valid0.gather(1, indices1)
                indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
                indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

                asc_order = torch.arange(self.num_traj).to('cuda')
                indices0_unique = indices0.unique()[1:] if -1 in indices0 else indices0.unique()
                combined = torch.cat((asc_order, indices0_unique))
                uniques, counts = combined.unique(return_counts=True)
                difference = uniques[counts == 1]
                indices0[indices0 == -1] = difference
                next_state_index = indices0[0]
            else:
                # bipartite graph matching via linear sum assignment. preferred as it is deterministic
                with torch.no_grad():
                    next_state_index = torch.as_tensor(linear_sum_assignment(cost_matrix.cpu())[1])
                
            matching_cost = matching_cost + nn.functional.pairwise_distance(next_state_pred, next_state[next_state_index], p=2)
            
            curr_state = next_state[next_state_index]
            
            # save and ensemble matched trajectory points 
            gen_traj_list.append(x[i_next][next_state_index])
        
        avg_matching_cost = matching_cost.mean() / x.shape[0]

        return rearrange(torch.dstack(gen_traj_list), 'n p t -> n t p'), avg_matching_cost

class TrajCritic(nn.Module):
    """Trajectory Critic in CATS (i.e., CatCrt)."""
    def __init__(self, num_traj=128, dim_loc=2, dim_embed=32, num_head=8):
        super(TrajCritic, self).__init__()
        
        self.num_traj = num_traj
        self.dim_loc = dim_loc
        self.dim_embed = dim_embed
        self.num_head = num_head
        
        # location embedding (2 -> 32)
        self.location_embedding = nn.Sequential(
            nn.Linear(self.dim_loc, self.dim_embed),
            nn.ReLU()
            )
        
        # time embedding (24 -> 32)
        self.time_embedding = nn.Sequential(
            nn.Embedding(24, self.dim_embed),
            nn.ReLU()
            )
        
        self.batch_norm = nn.BatchNorm1d(self.dim_embed * 2)
        
        # condition embedding
        self.condition_embedding = resnet_mini()
        self.tanh = nn.Tanh()

        # global spatial context attention
        self.global_sc_attn = nn.MultiheadAttention(self.dim_embed * 2, self.num_head)
        self.layer_norm = nn.LayerNorm(self.dim_embed * 2)
        
        # trajectory modeling
        self.traj_mod_gru = nn.GRUCell(self.dim_embed * 2, self.dim_embed * 2)
        
        # distance estimation head
        self.pred_head = nn.Sequential(
            nn.Linear(self.dim_embed * 4, self.dim_embed * 4),
            nn.ReLU(),
            nn.Linear(self.dim_embed * 4, 1)
            )

    def forward(self, x, cond):
        # location embedding 2->32
        embed_x = self.location_embedding(x).float()
        
        logits = torch.zeros((x.shape[0], self.dim_embed * 2)).to('cuda')
        
        # go through the sampled trajectory points along time
        for curr in range(x.shape[1]):
            embed_t = self.time_embedding(torch.as_tensor(curr).to('cuda')) * torch.ones((embed_x.shape[0],1)).to('cuda')
            embed_p = torch.cat([embed_x[:,curr,:], embed_t], dim = 1)
            embed_p = self.batch_norm(embed_p)

            # global spatial context attention for current-step trajectory points
            curr_attn_output, _ = self.global_sc_attn(embed_p, embed_p, embed_p)
            curr_state = self.layer_norm(embed_p + curr_attn_output)
            
            # trajectory modeling
            logits = self.traj_mod_gru(curr_state, logits)
        
        cond_embed = self.condition_embedding(cond)
        
        cond_logits = torch.cat([cond_embed, logits.mean(0).reshape(1,-1)], 1)
        
        return self.pred_head(cond_logits).reshape(1,-1)
    

class BasicBlock(nn.Module):
    """Basic Block for resnet."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

    
class ResNet(nn.Module):
    """Class for resnet."""
    def __init__(self, block, num_block):
        super().__init__()

        self.in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 48, num_block[0], 2)
        self.conv3_x = self._make_layer(block, 64, num_block[1], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)

        return output
    
def resnet_mini():
    return ResNet(BasicBlock, [2, 2])