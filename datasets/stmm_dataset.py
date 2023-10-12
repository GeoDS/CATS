from torch.utils.data import Dataset
from einops import rearrange
import numpy as np
import os


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
        
        return sampled_points / img.shape[1], real_points / img.shape[1], img

    
class Test_STMMDataset(Dataset):
    def __init__(self, sstm_path, sample_size):
        self.sstm_path = sstm_path
        self.sample_size = sample_size

    def __len__(self):
        return len(os.listdir(self.sstm_path))
    
    def sample_index(self, p):
        return np.dstack(np.unravel_index(np.random.choice(np.arange(p.size), size=self.sample_size, p=p.ravel()), p.shape))[0].astype(np.float)

    def __getitem__(self, idx):
        user_id = idx
        img = np.load(f'{self.sstm_path}/img_{user_id}.npy').astype(np.float)
        sampled_points = np.concatenate([self.sample_index(p.copy() / p.copy().sum()) for p in img])
        sampled_points = rearrange(sampled_points, '(n s) l -> n s l', s = self.sample_size)
        
        return user_id, sampled_points / img.shape[1], img
