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

def getDist(x1, y1, z1, x2, y2, z2):
    return np.sqrt((pow((x1-x2),2) + pow((y1-y2),2) + pow((z1-z2),2)))


if __name__ == '__main__':
    for _fin in os.listdir('/home/rdaneel/hri_costmap_traj'):
        if _fin.split('.')[-1] == 'csv':
            file_name = '/home/rdaneel/hri_costmap_traj/' + _fin
            traj_df = readFile(file_name)
            h_id = '1'
            r_id = '2'

            goal_pos2 = [traj_df['x'+r_id].tolist()[-1], traj_df['y'+r_id].tolist()[-1], traj_df['z'+r_id].tolist()[-1]]
            for idx, row in traj_df.iterrows():
                traj_df.loc[idx, 'f1'] = np.around(getDist(row['x'+r_id], row['y'+r_id], row['z'+r_id], row['x'+h_id], row['y'+h_id], row['z'+h_id]), 1)
                traj_df.loc[idx, 'f2'] = np.around(getDist(row['x'+r_id], row['y'+r_id], row['z'+r_id], goal_pos2[0], goal_pos2[1], goal_pos2[2]), 1)

            pre_row = pd.Series()
            for idx, row in traj_df.iterrows():
                if idx == 0:
                    pre_row = row
                else:
                    traj_df.loc[idx, 'a1'] = row['f1']-pre_row['f1']
                    traj_df.loc[idx, 'a2'] = row['f2']-pre_row['f2']
                    pre_row = row
                    
            traj_df.to_csv('/home/rdaneel/hri_costmap_traj/actions/' + _fin.split('.')[0] + '_action.csv')