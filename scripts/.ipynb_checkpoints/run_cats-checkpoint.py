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

from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset

from models.cats import TrajGenerator as CatGen
from models.cats import TrajCritic as CatCrt
from datasets.stmm_dataset import STMMDataset, Test_STMMDataset
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
    parser.add_argument("--sstm_path", type=str, default="./k_means_img")
    parser.add_argument("--weight_path", type=str, default="./cats_weights")
    parser.add_argument("--save_path", type=str, default="./cats_trajs")
    parser.add_argument("--sample_size", type=int, default=64)
    parser.add_argument("--manual_seed", type=int, default=2022)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()
    
    seed_everything(args.manual_seed)
    
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    
    test_data = Test_STMMDataset(args.sstm_path, args.sample_size)
    test_loader = DataLoader(dataset = test_data, batch_size=1, shuffle=False)
    
    trained_cats = torch.load(f'{args.weight_path}/G_epoch_{args.epoch}.pth')
    
    print("Running CATS...")
    
    for idx, data in enumerate(tqdm(test_loader)):
        user_id = data[0]
        sampled_points = data[1].to(args.device)[0].float()
        cats_trajs, _ = trained_cats(sampled_points)
        np.save(f'{args.save_path}/cats_{int(user_id)}.npy', cats_trajs.detach().to('cpu').numpy())
        
    print("CATS ran.")