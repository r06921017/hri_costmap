#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import time

import numpy as np
import pandas as pd
import os
import typing
from collections import namedtuple
import gym
from scipy.sparse import csr_matrix


class CostmapIRL:
    def __init__(self, relative2goal=True):
        self.csv_path = '/home/rdaneel/hri_costmap_traj/actions/'
        self.dp_list = list()
        self.file_name = list()
        self.goal_pos1 = list()
        self.goal_pos2 = list()

        for _fin in os.listdir(self.csv_path):
            self.dp_list.append(pd.read_csv(self.csv_path + _fin))
            self.file_name.append(_fin)
        
        # Get the maximum position among all demos
        tmp_f1 = list()
        tmp_f2 = list()
        tmp_a1 = list()
        tmp_a2 = list()
        for _dp in self.dp_list:
            tmp_f1.append(np.nanmax(_dp['f1'].tolist()))
            tmp_f2.append(np.nanmax(_dp['f2'].tolist()))
            tmp_a1.append(np.nanmax(_dp['a1'].tolist()))
            tmp_a2.append(np.nanmax(_dp['a2'].tolist()))

        self.max_f1 = np.around(np.nanmax(tmp_f1), 1)
        self.max_f2 = np.around(np.nanmax(tmp_f2), 1)
        self.max_a1 = np.around(np.nanmax(tmp_a1), 1)
        self.max_a2 = np.around(np.nanmax(tmp_a2), 1)

        # Get the minimum position among all demos
        tmp_f1 = list()
        tmp_f2 = list()
        tmp_a1 = list()
        tmp_a2 = list()
        for _dp in self.dp_list:
            tmp_f1.append(np.nanmin(_dp['f1'].tolist()))
            tmp_f2.append(np.nanmin(_dp['f2'].tolist()))
            tmp_a1.append(np.nanmin(_dp['a1'].tolist()))
            tmp_a2.append(np.nanmin(_dp['a2'].tolist()))

        self.min_f1 = np.around(np.nanmin(tmp_f1),1)
        self.min_f2 = np.around(np.nanmin(tmp_f2),1)
        self.min_a1 = np.around(np.nanmin(tmp_a1),1)
        self.min_a2 = np.around(np.nanmin(tmp_a2),1)

        print('max: ({0}, {1}, {2}, {3})'.format(self.max_f1, self.max_f2, self.max_a1, self.max_a2))
        print('min: ({0}, {1}, {2}, {3})'.format(self.min_f1, self.min_f2, self.min_a1, self.min_a2))

        # viewing manhattan distance of robot to goal and robot to human as states
        self.state_size = int(np.around((self.max_f1-self.min_f1)*(self.max_f2-self.min_f2)*100))
        print('state space size: {0}'.format(self.state_size))

        self.action_idx = {}
        for i, action in enumerate(np.arange(min(self.min_a1, self.min_a2), max(self.max_a1, self.max_a2) + 0.1, 0.1)):
            self.action_idx[i] = np.around(action, 1)

        self.inv_action_idx = {_tmp_v: _tmp_k for _tmp_k, _tmp_v in self.action_idx.items()}

    def jointAction2Idx(self, h_act, r_act):
        return self.action_idx[h_act] * len(self.action_idx) + self.action_idx[r_act]

    def idx2JointAction(self, idx):
        h_act_idx = idx // len(self.action_idx)
        r_act_idx = idx % len(self.action_idx)
        return self.inv_action_idx[h_act_idx], self.inv_action_idx[r_act_idx]

    def _pos2state(self, x, y, z):
        tmp_x = int(np.around(x, 1) * 10)
        tmp_y = int(np.around(y, 1) * 10)
        tmp_z = int(np.around(z, 1) * 10)
        tmp_x_length = int(np.around(self.x_length, 1) * 10)
        tmp_y_length = int(np.around(self.y_length, 1) * 10)
        return int((tmp_z * tmp_x_length * tmp_y_length) + (tmp_y * tmp_x_length) + tmp_x)


    def _state2pos(self, idx):
        z = int(idx // (self.x_length * self.y_length))
        idx -= (z * self.x_length * self.y_length)
        y = int(idx // self.x_length)
        x = int(idx % self.x_length)

        x = np.around(x / 10., 1)
        y = np.around(y / 10., 1)
        z = np.around(z / 10., 1)
        return (x, y, z);

    def pair_transition(self, h_obs, r_obs, h_act, r_act, h_next_obs, r_next_obs, goal_pos):
        (hx, hy, hz) = _state2pos(h_obs, self.x_length, self.y_length)
        h_nx = hx + self.inv_action_idx[h_act][0]
        h_ny = hy + self.inv_action_idx[h_act][1]
        h_nz = hz + self.inv_action_idx[h_act][2]

        # if self.min_x <= h_nx <= self.max_x and self.min_y <= h_ny <= self.max_y and self.min_z <= h_nz <= self.max_z:
        #     h_ns = _pos2state(h_nx, h_ny, h_nz)
        # else:
        #     h_ns = _pos2state(hx, hy, hz)
        
        (rx, ry, rz) = _state2pos(r_obs, self.x_length, self.y_length)
        r_nx = rx + self.inv_action_idx[r_act][0]
        r_ny = ry + self.inv_action_idx[r_act][1]
        r_nz = rz + self.inv_action_idx[r_act][2]

        # if self.min_x <= r_nx <= self.max_x and self.min_y <= r_ny <= self.max_y and self.min_z <= r_nz <= self.max_z:
        #     r_ns = _pos2state(r_nx, r_ny, r_nz)
        # else:
        #     r_ns = _pos2state(rx, ry, rz)


        dis2goal = self.getdist(r_obs, goal_pos)
        dis2human = self.getdist(r_obs, h_obs)

        # Deterministic, probability = 1
        if h_next_obs == h_ns and r_next_obs == r_ns:
            return 1.
        else:
            return 0.

    def extract_features(self, h_pos, r_pos, goal_pos):
        """[summary]
        manhattan distance to goal and distance to human from robot end effector position
        Arguments:
            h_pos {Tuple(int, int, int)} -- [description]
            r_pos {[type]} -- [description]
            goal_pos {[type]} -- [description]
        """
        dist2goal = abs(goal_pos[0] - r_pos[0]) + abs(goal_pos[1] - r_pos[1])
        dist2human = abs(h_pos[0] - r_pos[0]) + abs(h_pos[1] - r_pos[1])
        return dist2goal, dist2human

    def observation_space(self):
        """[summary]
        Using manhattan distance as state space
        Returns:
            [type] -- [description]
        """
        return gym.spaces.Discrete(self.state_size)

    def action_space(self):
        return gym.spaces.Discrete(len(self.action_idx)**2)

    def main(self):
        
        return
    
if __name__ == '__main__':
    costmap_irl = CostmapIRL()