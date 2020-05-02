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
        
        for _dp in self.dp_list:
            # get position of last row
            self.goal_pos1 = [_dp['x1'].tolist()[-1], _dp['y1'].tolist()[-1], _dp['z1'].tolist()[-1]]
            self.goal_pos2 = [_dp['x2'].tolist()[-1], _dp['y2'].tolist()[-1], _dp['z2'].tolist()[-1]]

            if relative2goal:
                # position relative to the goal
                _dp['x1'] -= self.goal_pos2[0]
                _dp['y1'] -= self.goal_pos2[1]
                _dp['z1'] -= self.goal_pos2[2]

                _dp['x2'] -= self.goal_pos2[0]
                _dp['y2'] -= self.goal_pos2[1]
                _dp['z2'] -= self.goal_pos2[2]

        # Get the maximum position among all demos
        tmp_x = list()
        tmp_y = list()
        tmp_z = list()
        for _dp in self.dp_list:
            tmp_x.append(np.around(max(max(_dp['x1'].tolist()), max(_dp['x2'].tolist())), 1))
            tmp_y.append(np.around(max(max(_dp['y1'].tolist()), max(_dp['y2'].tolist())), 1))
            tmp_z.append(np.around(max(max(_dp['z1'].tolist()), max(_dp['z2'].tolist())), 1))

        self.max_x = max(tmp_x)
        self.max_y = max(tmp_y)
        self.max_z = max(tmp_z)

        # Get the minimum position among all demos
        tmp_x = list()
        tmp_y = list()
        tmp_z = list()
        for _dp in self.dp_list:
            tmp_x.append(np.around(min(min(_dp['x1'].tolist()), min(_dp['x2'].tolist())), 1))
            tmp_y.append(np.around(min(min(_dp['y1'].tolist()), min(_dp['y2'].tolist())), 1))
            tmp_z.append(np.around(min(min(_dp['z1'].tolist()), min(_dp['z2'].tolist())), 1))

        self.min_x = min(tmp_x)
        self.min_y = min(tmp_y)
        self.min_z = min(tmp_z)

        print('max: ({0}, {1}, {2})'.format(self.max_x, self.max_y, self.max_z))
        print('min: ({0}, {1}, {2})'.format(self.min_x, self.min_y, self.min_z))

        self.x_length = np.around(self.max_x - self.min_x, 1)
        self.y_length = np.around(self.max_y - self.min_y, 1)
        self.z_length = np.around(self.max_z - self.min_z, 1)

        # viewing manhattan distance of robot to goal and robot to human as states
        self.state_size = int(np.around(((self.x_length + self.y_length + self.z_length)*10) ** 2))
        print('state space size: {0}'.format(self.state_size))

        self.action_idx = {
            (0, 0, 0): 0,
            (0.1, 0, 0): 1,
            (-0.1, 0, 0): 2,
            (0, 0.1, 0): 3,
            (0, -0.1, 0): 4,
            (0, 0, 0.1): 5,
            (0, 0, -0.1): 6,

            (0.1, 0.1, 0): 7,
            (-0.1, 0.1, 0): 8,
            (0.1, -0.1, 0): 9,
            (-0.1, -0.1, 0): 10,
            
            (0.1, 0, 0.1): 11,
            (-0.1, 0, 0.1): 12,
            (0.1, 0, -0.1): 13,
            (-0.1, 0, -0.1): 14,

            (0, 0.1, 0.1): 15,
            (0, -0.1, 0.1): 16,
            (0, 0.1, -0.1): 17,
            (0, -0.1, -0.1): 18,

            (0.1, 0.1, 0.1): 19,
            (-0.1, 0.1, 0.1): 20,
            (0.1, -0.1, 0.1): 21,
            (-0.1, -0.1, 0.1): 22,
            (0.1, 0.1, -0.1): 23,
            (-0.1, 0.1, -0.1): 24,
            (0.1, -0.1, -0.1): 25,
            (-0.1, -0.1, -0.1): 26
        }
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

    def pair_transition(self, h_obs, r_obs, h_act, r_act, h_next_obs, r_next_obs):
        (hx, hy, hz) = _state2pos(h_obs, self.x_length, self.y_length)
        h_nx = hx + self.inv_action_idx[h_act][0]
        h_ny = hy + self.inv_action_idx[h_act][1]
        h_nz = hz + self.inv_action_idx[h_act][2]

        if self.min_x <= h_nx <= self.max_x and self.min_y <= h_ny <= self.max_y and self.min_z <= h_nz <= self.max_z:
            h_ns = _pos2state(h_nx, h_ny, h_nz)
        else:
            h_ns = _pos2state(hx, hy, hz)
        
        (rx, ry, rz) = _state2pos(r_obs, self.x_length, self.y_length)
        r_nx = rx + self.inv_action_idx[r_act][0]
        r_ny = ry + self.inv_action_idx[r_act][1]
        r_nz = rz + self.inv_action_idx[r_act][2]

        if self.min_x <= r_nx <= self.max_x and self.min_y <= r_ny <= self.max_y and self.min_z <= r_nz <= self.max_z:
            r_ns = _pos2state(r_nx, r_ny, r_nz)
        else:
            r_ns = _pos2state(rx, ry, rz)

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
        return gym.spaces.Discrete()

    def action_space(self):
        return gym.spaces.Discrete(len(self.action_idx)**2)

    def main(self):
        
        return
    
if __name__ == '__main__':
    costmap_irl = CostmapIRL()