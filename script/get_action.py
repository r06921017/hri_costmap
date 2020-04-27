#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import os
import logging

def readFile(f_name):
    if os.path.exists(f_name):
        return pd.read_csv(f_name)
    else:
        logging.error('{0} does not exist!'.format(f_name))
        exit(1)

if __name__ == '__main__':
    file_name = '/home/rdaneel/hri_costmap_traj/S032C003P105R002A107.csv'
    traj_df = readFile(file_name)

    pre_row = pd.Series()
    for idx, row in traj_df.iterrows():
        if idx == 0:
            pre_row = row
        else:
            a1 = (row['x1']-pre_row['x1'], row['y1']-pre_row['y1'], row['z1']-pre_row['z1'])
            a2 = (row['x2']-pre_row['x2'], row['y2']-pre_row['y2'], row['z2']-pre_row['z2'])
            traj_df.loc[idx, 'ax1'] = np.around(row['x1']-pre_row['x1'], 2)
            traj_df.loc[idx, 'ay1'] = np.around(row['y1']-pre_row['y1'], 2)
            traj_df.loc[idx, 'az1'] = np.around(row['z1']-pre_row['z1'], 2)
            traj_df.loc[idx, 'ax2'] = np.around(row['x2']-pre_row['x2'], 2)
            traj_df.loc[idx, 'ay2'] = np.around(row['y2']-pre_row['y2'], 2)
            traj_df.loc[idx, 'az2'] = np.around(row['z2']-pre_row['z2'], 2)
            pre_row = row
            
    traj_df.to_csv('/home/rdaneel/hri_costmap_traj/actions/S032C003P105R002A107.csv')