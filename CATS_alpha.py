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

import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from einops import rearrange
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment


class logger(object):
    def __init__(self, logging_path):
        self.info_log = open(f"{logging_path}/info_log.txt", "a")
        self.info_log.flush()
        
        self.loss_log = open(f"{logging_path}/loss_log.csv", "a")
        self.loss_log.write('Epoch,Iteration,Loss_D,Loss_G,D_x,D_G_z1,D_G_z2\n')
        self.loss_log.flush()
    
    def info(self, msg):
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f'[{curr_time}] {msg}'
        print(msg)
        self.info_log.write(msg + '\n')
        self.info_log.flush()
        
    def update_loss(self, msg):
        self.loss_log.write(msg)
        self.loss_log.flush()
        
    def close(self):
        self.loss_log.close()
        self.info_log.close()
        


class STMMDataset(Dataset):
    def __init__(self, user_id_list, img_path, traj_path, sample_size):
        self.user_id_list = user_id_list
        self.img_path = img_path
        self.traj_path = traj_path
        self.sample_size = sample_size

    def __len__(self):
        self.filelength = len(self.user_id_list)
        return self.filelength
    
    def sample_index(self, p):
        return np.dstack(np.unravel_index(np.random.choice(np.arange(p.size), size=self.sample_size, p=p.ravel()), p.shape))[0].astype(np.float)

    def __getitem__(self, idx):
        user_id = self.user_id_list[idx]
        img = np.load(f'{self.img_path}/img_{user_id}.npy').astype(np.float)
        real_points = np.load(f'{self.traj_path}/traj_{user_id}.npy').astype(np.float)
        
        sampled_points = np.concatenate([self.sample_index(p.copy() / p.copy().sum()) for p in img])
        sampled_points = rearrange(sampled_points, '(n s) l -> n s l', s = self.sample_size)
        
        return sampled_points / 127.0, real_points / 127.0


class BipartiteGraphMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        
        
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
            iters=10000)
        return scores

    
# Trajectory Generator
class TrajGenerator(nn.Module):
    def __init__(self, num_traj=128, dim_loc=2, dim_embed=32, num_head=8, drop_out_rate=0.5):
        super(TrajGenerator, self).__init__()
        
        self.num_traj = num_traj
        self.dim_loc = dim_loc
        self.dim_embed = dim_embed
        self.num_head = num_head
        self.drop_out_rate = drop_out_rate
        
        # location embedding (2 -> 32)
        self.location_embedding = nn.Sequential(
            nn.Linear(self.dim_loc, self.dim_embed),
            nn.LeakyReLU(0.2)
            )
        
        # time embedding (24 -> 32)
        self.time_embedding = nn.Sequential(
            nn.Embedding(24, self.dim_embed),
            nn.LeakyReLU(0.2)
            )
        
#         # spatio-temporal fusion (64 -> 64)
#         self.fusion_layer = nn.Sequential(
#             nn.Linear(self.dim_embed * 2, self.dim_embed * 2),
#             nn.LeakyReLU(0.2)
#             )
        
        # global spatial context attention
        self.global_sc_attn = nn.MultiheadAttention(self.dim_embed * 2, self.num_head)
        self.layer_norm = nn.LayerNorm(self.dim_embed * 2)
        self.drop_out = nn.Dropout(self.drop_out_rate)
        
        # trajectory modeling
        self.traj_mod_gru = nn.GRUCell(self.dim_embed * 2, self.dim_embed * 2)
        self.tanh = nn.Tanh()
        
        # bipartite graph matching
#         self.bipartite_graph_matcher = BipartiteGraphMatcher()

    def forward(self, x):
        # location embedding
        embed_x = self.location_embedding(x).float()
        
        # time embedding
        embed_t = self.time_embedding(torch.as_tensor(0).to('cuda')) * torch.ones((self.num_traj,1)).to('cuda')
        
        # spatio-temporal concatenation
        embed_p = torch.cat([embed_x[0], embed_t], dim = 1)

        # spatio-temporal fusion
#         embed_p = self.fusion_layer(embed_p)

        # global spatial context attention for current-step trajectory points
        curr_attn_output, _ = self.global_sc_attn(embed_p, embed_p, embed_p)
        # curr_attn_output = self.drop_out(curr_attn_output)
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
            
            # spatio-temporal fusion
#             next_embed_p = self.fusion_layer(next_embed_p)
            
            # global spatial context attention for next-step trajectory points
            next_attn_output, _ = self.global_sc_attn(next_embed_p, next_embed_p, next_embed_p)
            # next_attn_output = self.drop_out(next_attn_output)
            next_state = self.layer_norm(next_embed_p + next_attn_output)
            # normalize the value range to (-1, 1)
            next_state = self.tanh(next_state)
            
            # the prediction of next-step trajectory points. value in (-1, 1)
            next_state_pred = self.traj_mod_gru(curr_state, next_state_pred)
            
            # cost matrix based on L2 cost
            cost_matrix = torch.cdist(next_state_pred, next_state, p=2)
            # cost_matrix = 1.00001 - pairwise_cosine_similarity(next_state_pred, next_state)
            
#             # bipartite graph matching between the next state prediction and next state ground truth
#             matching_res = self.bipartite_graph_matcher(cost_matrix.unsqueeze(0).log_softmax(dim=-2))
            
#             # Get the successful matches
#             max0, max1 = matching_res[:, :-1, :-1].max(2), matching_res[:, :-1, :-1].max(1)
#             indices0, indices1 = max0.indices, max1.indices
#             mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
#             mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
#             zero = scores.new_tensor(0)
#             mscores0 = torch.where(mutual0, max0.values.exp(), zero)
#             mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
#             valid0 = mutual0 & (mscores0 > 0.)
#             valid1 = mutual1 & valid0.gather(1, indices1)
#             indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
#             indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

            # bipartite graph matching between the next state prediction and next state ground truth
            with torch.no_grad():
                next_state_index = torch.as_tensor(linear_sum_assignment(cost_matrix.cpu())[1])
            
            # with torch.no_grad():
                # cosine_cost = cosine_cost - nn.functional.cosine_similarity(hx, next_state) + 1
                # cosine_cost = cosine_cost + nn.functional.pairwise_distance(hx, next_state, p=2)
                
            matching_cost = matching_cost + nn.functional.pairwise_distance(next_state_pred, next_state[next_state_index], p=2)
            
            curr_state = next_state[next_state_index]
            
            # save and ensemble matched trajectory points 
            gen_traj_list.append(x[i_next][next_state_index])
        
        avg_matching_cost = matching_cost.mean() / x.shape[0]

        return rearrange(torch.dstack(gen_traj_list), 'n p t -> n t p'), avg_matching_cost
    

class TrajDiscriminator(nn.Module):
    def __init__(self, num_traj=128, dim_loc=2, dim_embed=32):
        super(TrajDiscriminator, self).__init__()
        
        self.num_traj = num_traj
        self.dim_loc = dim_loc
        self.dim_embed = dim_embed
        
        # location embedding (2 -> 32)
        self.location_embedding = nn.Sequential(
            nn.Linear(self.dim_loc, self.dim_embed),
            nn.LeakyReLU(0.2)
            )
        
        # time embedding (24 -> 32)
        self.time_embedding = nn.Sequential(
            nn.Embedding(24, self.dim_embed),
            nn.LeakyReLU(0.2)
            )
        
#         # spatio-temporal fusion (64 -> 64)
#         self.fusion_layer = nn.Sequential(
#             nn.Linear(self.dim_embed * 2, self.dim_embed * 2),
#             nn.LeakyReLU(0.2)
#             )
        
        # trajectory modeling
        self.traj_mod_gru = nn.GRUCell(self.dim_embed * 2, self.dim_embed * 2)
        
        # prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(self.dim_embed * 2, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        # location embedding 2->32
        embed_x = self.location_embedding(x).float()
        
        logits = torch.zeros((x.shape[0], self.dim_embed * 2)).to('cuda')
        
        # go through the sampled trajectory points along time
        for curr in range(x.shape[1]):
            embed_t = self.time_embedding(torch.as_tensor(curr).to('cuda')) * torch.ones((embed_x.shape[0],1)).to('cuda')
            embed_p = torch.cat([embed_x[:,curr,:], embed_t], dim = 1)
#             embed_p = self.fusion_layer(embed_p)
            
            # trajectory modeling
            logits = self.traj_mod_gru(embed_p, logits)
        
        return self.pred_head(logits).flatten()
    
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="../encoded_img_24")
    parser.add_argument("--traj_path", type=str, default="../encoded_traj_24")
    parser.add_argument("--save_path", type=str, default="../weights_exp/weights_weighted_loss")
    parser.add_argument("--manual_seed", type=int, default=2022)
    parser.add_argument("--sample_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dim_embed", type=int, default=32)
    parser.add_argument("--dim_loc", type=int, default=2)
    parser.add_argument("--lr_d", type=float, default=0.005)
    parser.add_argument("--lr_g", type=float, default=0.005)
    parser.add_argument("--drop_out_rate", type=float, default=0.5)
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    
    # initialize logger
    logger = logger(args.save_path)
    
    logger.info(f"PARAMS: {vars(args)}")
    
    seed_everything(args.manual_seed)
    
    num_user = len(os.listdir(args.img_path))
    
    train_list, valid_list = train_test_split(list(range(num_user)), 
                                          test_size=args.test_rate,
                                          random_state=args.manual_seed)
    
    logger.info(f"Train Data: {len(train_list)}")
    logger.info(f"Validation Data: {len(valid_list)}")
    train_data = STMMDataset(train_list, args.img_path, args.traj_path, args.sample_size)
    valid_data = STMMDataset(valid_list, args.img_path, args.traj_path, args.sample_size)
    train_loader = DataLoader(dataset = train_data, batch_size=1, shuffle=False)
    valid_loader = DataLoader(dataset = valid_data, batch_size=1, shuffle=False)
    
    # Create the Trajectory Genaerator
    traj_generator = TrajGenerator(num_traj=args.sample_size,
                                   dim_loc=args.dim_loc,
                                   dim_embed=args.dim_embed,
                                   num_head=args.num_heads,
                                   drop_out_rate=args.drop_out_rate).to(args.device)
    
    # Create the Trajectory Generator
    traj_discriminator = TrajDiscriminator(num_traj=args.sample_size,
                                           dim_loc=args.dim_loc,
                                           dim_embed=args.dim_embed).to(args.device)
    
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerDiscriminator = optim.Adam(traj_discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))
    optimizerGenerator = optim.Adam(traj_generator.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
    
    # Training Loop
    iters = 0

    logger.info("Starting Training Loop...")
    
    # For each epoch
    for epoch in range(args.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_loader):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            traj_discriminator.zero_grad()
            # Format batch
            sampled_points = data[0].to(args.device)[0].float()
            real_points = data[1].to(args.device)[0].float()

            real_sample_size = real_points.shape[0]
            real_labels = torch.full((real_sample_size,), 1., dtype=torch.float, device=args.device)

            # Forward pass real batch through D
            output = traj_discriminator(real_points)
            # Calculate loss on all-real batch
            errD_real = criterion(output, real_labels)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate fake image batch with G

            fake_sample_size = sampled_points.shape[1]
            fake_labels = torch.full((fake_sample_size,), 0., dtype=torch.float, device=args.device)

            fake_trajs, matching_cost = traj_generator(sampled_points)

            # Classify all fake batch with D
            output = traj_discriminator(fake_trajs.detach())
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, fake_labels)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerDiscriminator.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            traj_generator.zero_grad()
            real_labels = torch.full((fake_sample_size,), 1.0, dtype=torch.float, device=args.device)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = traj_discriminator(fake_trajs)
            # Calculate G's loss based on this output
            errG = criterion(output, real_labels)
            
            with torch.no_grad():
                matching_cost *= 0.1
                matching_cost += errG
            
#           # Calculate gradients for G
            # errG.backward()
            matching_cost.backward()
        
            D_G_z2 = output.mean().item()
#             # Update G
            optimizerGenerator.step()

            # Output training stats
            if i % 50 == 0:
                logger.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.num_epochs - 1, i, len(train_loader),
                         errD.item(), matching_cost.item(), D_x, D_G_z1, D_G_z2))

            logger.update_loss(f'{epoch},{iters},{errD.item()},{matching_cost.item()},{D_x},{D_G_z1},{D_G_z2}\n')
            iters += 1
            
        torch.save(traj_generator, f'{args.save_path}/G_epoch_{epoch}.pth')
        torch.save(traj_discriminator, f'{args.save_path}/D_epoch_{epoch}.pth')
    logger.close()