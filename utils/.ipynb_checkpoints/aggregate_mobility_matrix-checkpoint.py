import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./mocked_individual_gps_data.csv")
    parser.add_argument("--save_path", type=str, default="./stmm_data")
    parser.add_argument("--agg_size", type=int, default=128)
    args = parser.parse_args()
    
    traj_df = pd.read_csv(args.csv_path)
    
    lat_extent = traj_df['Lat of Visit'].max() - traj_df['Lat of Visit'].min()
    lon_extent = traj_df['Lon of Visit'].max() - traj_df['Lon of Visit'].min()
    
    if lat_extent > lon_extent:
        delta = lat_extent / args.agg_size
        traj_df['agg_x'] = (traj_df['Lat of Visit'] - traj_df['Lat of Visit'].min()) // delta
        traj_df['agg_y'] = (traj_df['Lon of Visit'] - traj_df['Lon of Visit'].min() - (lat_extent - lon_extent) / 2) // delta
    else:
        delta = lon_extent / args.agg_size
        traj_df['agg_x'] = (traj_df['Lat of Visit'] - traj_df['Lat of Visit'].min() - (lon_extent - lat_extent) / 2) // delta
        traj_df['agg_y'] = (traj_df['Lon of Visit'] - traj_df['Lon of Visit'].min()) // delta
        
    for u_id, u_df in tqdm(traj_df.groupby(['user_id'])):
        u_agg_mob_mat = np.zeros((24, args.agg_size, args.agg_size))
        for hour, h_df in u_df.groupby(['Hour']):
            for agg_xy, xy_df in h_df.groupby(['agg_x', 'agg_y']):
                agg_x = int(agg_xy[0]) if int(agg_xy[0]) <= (args.agg_size - 1) else (args.agg_size - 1)
                agg_y = int(agg_xy[1]) if int(agg_xy[1]) <= (args.agg_size - 1) else (args.agg_size - 1)
                value = len(xy_df)
                u_agg_mob_mat[hour][(agg_x, agg_y)] = value
            u_agg_mob_mat[hour] /= len(h_df)
        np.save(f'{args.save_path}/mock_{u_id}.npy', u_agg_mob_mat)
