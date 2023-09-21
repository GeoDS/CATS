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

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from einops import rearrange

from scipy.optimize import linear_sum_assignment

from models.cats import TrajGenerator as CatGen
from models.cats import TrajCritic as CatCrt
from datasets.stmm_dataset import STMMDataset
from utils.log_util import logger

        
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
    parser.add_argument("--img_path", type=str, default="./data/original_mobility_marix")
    parser.add_argument("--traj_path", type=str, default="./data/original_trajectory")
    parser.add_argument("--save_path", type=str, default="./cats_weights")
    parser.add_argument("--manual_seed", type=int, default=2022)
    parser.add_argument("--sample_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dim_embed", type=int, default=32)
    parser.add_argument("--dim_loc", type=int, default=2)
    parser.add_argument("--num_loop_d", type=int, default=1)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--use_sinkhorn", action="store_true")
    parser.add_argument("--sinkhorn", type=float, default=0)
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=int, default=-1)
    args = parser.parse_args()
    
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    
    # initialize logger
    logger = logger(args.save_path)
    
    logger.info(f"PARAMS: {vars(args)}")
    
    seed_everything(args.manual_seed)
    
    num_user = len(os.listdir(args.img_path))
    
    train_list, _ = train_test_split(list(range(num_user)), 
                                          test_size=args.test_rate,
                                          random_state=args.manual_seed)
    
    logger.info(f"Train Data: {len(train_list)}")
    train_data = STMMDataset(train_list, args.img_path, args.traj_path, args.sample_size)
    train_loader = DataLoader(dataset = train_data, batch_size=1, shuffle=False)
    
    # Create the Trajectory Generator and Critic
    if args.resume < 0:
        cat_gen = CatGen(num_traj=args.sample_size,
                       dim_loc=args.dim_loc,
                       dim_embed=args.dim_embed,
                       num_head=args.num_heads,
                       sinkhorn=args.sinkhorn,
                       use_sinkhorn=args.use_sinkhorn).to(args.device)
        
        cat_crt = CatCrt(num_traj=args.sample_size,
                                               dim_loc=args.dim_loc,
                                               dim_embed=args.dim_embed).to(args.device)
    else:
        cat_gen = torch.load(f'{args.save_path}/G_epoch_{args.resume}.pth')
        cat_crt = torch.load(f'{args.save_path}/D_epoch_{args.resume}.pth')

    # Setup for both G and D
    CatCrt_optimizer = optim.RMSprop(cat_crt.parameters(), lr=args.lr_d, eps=1e-5)
    CatGen_optimizer = optim.RMSprop(cat_gen.parameters(), lr=args.lr_g, eps=1e-5)

    logger.info("Training CATS...")
    
    # For each epoch
    iters = 0
    for epoch in range(args.resume+1, args.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_loader):
            for _ in range(args.num_loop_d):
                CatCrt_optimizer.zero_grad()
                
                # Format batch
                sampled_points = data[0].to(args.device)[0].float()
                real_points = data[1].to(args.device)[0].float()
                cond = data[2].to(args.device).float()

                # Forward pass real batch through D
                output_real = cat_crt(real_points.detach(), cond.detach())
                D_x = output_real.mean().item()
                
                # generate synthetic trajs
                fake_trajs, matching_cost = cat_gen(sampled_points)
                
                # Classify all fake batch with D
                output_fake = cat_crt(fake_trajs.detach(), cond.detach())
                D_G_z1 = output_fake.mean().item()
                
                # loss, which is also negative W-dist
                errD = torch.mean(output_fake) - torch.mean(output_real)
                errD.backward()

                # Update D
                CatCrt_optimizer.step()
                for p in cat_crt.parameters():
                    p.data.clamp_(-0.1, 0.1)
                    
            CatGen_optimizer.zero_grad()
            
            # Magicbox trick
            # https://arxiv.org/abs/1802.05098
            fake_trajs_diff = matching_cost + (fake_trajs - matching_cost).detach()
            
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output_fake = cat_crt(fake_trajs_diff, cond)
            D_G_z2 = output_fake.mean().item()
            
            # Calculate G's loss with matching cost as constraint
            errG = -torch.mean(output_fake) + 0.01 * matching_cost
            
            # Calculate gradients for G
            errG.backward()
    
            # Update G
            CatGen_optimizer.step()

            # Output training stats
            if i % 50 == 0:
                logger.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tmatching_cost: %.4f'
                      % (epoch, args.num_epochs - 1, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, matching_cost.item()))

            logger.update_loss(f'{epoch},{iters},{errD.item()},{errG.item()},{D_x},{D_G_z1},{D_G_z2}\n')
            iters += 1
            
        torch.save(cat_gen, f'{args.save_path}/G_epoch_{epoch}.pth')
        torch.save(cat_crt, f'{args.save_path}/D_epoch_{epoch}.pth')
        logger.info(f"CATS weights at epoch {epoch} are saved to {args.save_path}.")
    logger.info("CATS Training Finished.")
    logger.close()