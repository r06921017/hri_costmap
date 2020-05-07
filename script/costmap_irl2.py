#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import gym
import time
import typing
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.sparse import csr_matrix

from max_ent_irl import MaxEntIRL
from util.plotting import plot_grid_map, plot_policy, plot_dataset_distribution
from util.reward import construct_goal_reward, construct_human_radius_reward
from util.dataset import Dataset, transition


class CostmapIRL:
    def __init__(self):
        self.csv_path = '/home/rdaneel/hri_costmap_traj/A107_hp/actions/'
        self.action_success_rate = 1.0
        self.df_list = list()
        self.file_name = list()
        self.goal_states = list()
        self.precision = 0.01
        self.around_digit = len(str(self.precision).split('.'))

        for _fin in os.listdir(self.csv_path):
            self.df_list.append(pd.read_csv(self.csv_path + _fin))
            self.file_name.append(_fin)

        # Get max length
        self.max_length = 0
        for _df in self.df_list:
            if _df.shape[0] > self.max_length:
                self.max_length = _df.shape[0]
        
        # Get the maximum position among all demos
        tmp_f1 = list()
        tmp_f2 = list()
        tmp_a1 = list()
        tmp_a2 = list()
        for i, _df in enumerate(self.df_list):
            tmp_f1.append(np.nanmax(_df['f1'].tolist()))
            tmp_f2.append(np.nanmax(_df['f2'].tolist()))
            tmp_a1.append(np.nanmax(_df['a1'].tolist()))
            tmp_a2.append(np.nanmax(_df['a2'].tolist()))

        self.max_f1 = np.around(np.nanmax(tmp_f1), self.around_digit)
        self.max_f2 = np.around(np.nanmax(tmp_f2), self.around_digit)
        self.max_a1 = np.around(np.nanmax(tmp_a1), self.around_digit)
        self.max_a2 = np.around(np.nanmax(tmp_a2), self.around_digit)

        # Get the minimum position among all demos
        tmp_f1 = list()
        tmp_f2 = list()
        tmp_a1 = list()
        tmp_a2 = list()
        for _df in self.df_list:
            tmp_f1.append(np.nanmin(_df['f1'].tolist()))
            tmp_f2.append(np.nanmin(_df['f2'].tolist()))
            tmp_a1.append(np.nanmin(_df['a1'].tolist()))
            tmp_a2.append(np.nanmin(_df['a2'].tolist()))

        self.min_f1 = np.around(np.nanmin(tmp_f1),self.around_digit)
        self.min_f2 = np.around(np.nanmin(tmp_f2),self.around_digit)
        self.min_a1 = np.around(np.nanmin(tmp_a1),self.around_digit)
        self.min_a2 = np.around(np.nanmin(tmp_a2),self.around_digit)

        self.f1_idx = dict()
        for i, f in enumerate(np.arange(self.min_f1, self.max_f1 + self.precision/2.0, self.precision)):
            self.f1_idx[i] = np.around(f, self.around_digit)
        self.f1_length = len(self.f1_idx)
        self.inv_f1_idx = {_tmp_v: _tmp_k for _tmp_k, _tmp_v in self.f1_idx.items()}

        self.f2_idx = dict()
        for i, f in enumerate(np.arange(self.min_f2, self.max_f2 + self.precision/2.0, self.precision)):
            self.f2_idx[i] = np.around(f, self.around_digit)
        self.f2_length = len(self.f2_idx)
        self.inv_f2_idx = {_tmp_v: _tmp_k for _tmp_k, _tmp_v in self.f2_idx.items()}

        print('max: ({0}, {1}, {2}, {3})'.format(self.max_f1, self.max_f2, self.max_a1, self.max_a2))
        print('min: ({0}, {1}, {2}, {3})'.format(self.min_f1, self.min_f2, self.min_a1, self.min_a2))

        # Get goal positions
        self.goal_states = []

        # viewing manhattan distance of robot to goal and robot to human as states
        self.state_size = self.f1_length * self.f2_length
        print('state space size: {0}'.format(self.state_size))

        self.action_idx = dict()
        for i, a1 in enumerate(np.arange(self.min_a1, self.max_a1 + self.precision/2.0, self.precision)):
            for j, a2 in enumerate(np.arange(self.min_a2, self.max_a2 + self.precision/2.0, self.precision)):
                a_id = i*len(np.arange(self.min_a2, self.max_a2 + self.precision/2.0, self.precision)) + j
                self.action_idx[a_id] = (np.around(a1, self.around_digit), np.around(a2, self.around_digit))

        print('action space size: {0}'.format(len(self.action_idx)))

        self.inv_action_idx = {_tmp_v: _tmp_k for _tmp_k, _tmp_v in self.action_idx.items()}

        S, A = self.observation_space().n, self.action_space().n
        self.T = [csr_matrix((S, S), dtype=np.float) for _ in range(A)]

    
    def _valid_pos(self, in_state):
        return in_state[0] in self.f1_idx.values() and in_state[1] in self.f2_idx.values()

    def _construct_transition_fn(self):
        # T[a][s, s'] implemented as a length |A| list of sparse SxS matrices
        S, A = self.observation_space().n, self.action_space().n
        T = [csr_matrix((S, S), dtype=np.float) for _ in range(A)]

        for i in self.f1_idx.keys():
            for j in self.f2_idx.keys():
                cur_s = [self.f1_idx[i], self.f2_idx[j]]
                next_sid = list()
                for a_id in self.action_idx.keys():
                    is_val = True
                    neighbor = [cur_s[0] + self.action_idx[a_id][0], cur_s[1] + self.action_idx[a_id][1]]
                    if not self._valid_pos(neighbor):  # not valid
                        is_val = False
                        neighbor = cur_s

                    # add (action, next state, is_valid)
                    next_sid.append((a_id, self._pos2state(neighbor[0], neighbor[1]), is_val))

                # uniform weighting of all other valid transitions
                fail_prob = (1.0 - self.action_success_rate)
                if len(next_sid) > 0:
                    fail_prob /= len(next_sid) - 1

                for _ns in next_sid:
                    if _ns[2]:
                        T[_ns[0]][self._pos2state(cur_s[0], cur_s[1]), _ns[1]] = self.action_success_rate

                    else:
                        T[_ns[0]][self._pos2state(cur_s[0], cur_s[1]), _ns[1]] = fail_prob

        # Block transition out of goal positions
        for goal_state in self.goal_states:
            sprime_prob = np.zeros((S,))
            sprime_prob[goal_state] = 1
            for a in self.action_idx.keys():
                T[a][goal_state, :] = sprime_prob

        return T


    def _pos2state(self, f1, f2):
        return int(self.inv_f1_idx[f1]*self.f2_length + self.inv_f2_idx[f2])

    def _state2pos(self, idx):
        f1 = self.f1_idx[idx // self.f2_length]
        f2 = self.f2_idx[idx % self.f2_length]

        return f1, f2;

    def _transition(self, obs, in_action, next_obs):
        cur_f1, cur_f2 = self._state2pos(obs)
        act1 = self.action_idx[in_action][0]
        act2 = self.action_idx[in_action][1]

        tmp_f1 = cur_f1 + act1
        tmp_f2 = cur_f2 + act2

        next_f1, next_f2 = self._state2pos(next_obs)
        
        # Deterministic, probability = 1
        if tmp_f1 == next_f1 and tmp_f2 == next_f2:
            return 1.
        else:
            return 0.

    def observation_space(self):
        return gym.spaces.Discrete(self.state_size)

    def action_space(self):
        return gym.spaces.Discrete(len(self.action_idx))

    def _feature_map(self, state):
        # TODO: move out of gridworld
        feat = np.zeros(self.observation_space().n)
        feat[state] = 1
        return feat

    
    def collect_demo(self):        
        dataset = Dataset(self.max_length)
        for idx, _df in enumerate(self.df_list):
            t = list()
            for idx in range(_df.shape[0]-1):  # before termination?
                t.append(transition(
                    obs=self._pos2state(np.around(_df['f1'][idx], self.around_digit), np.around(_df['f2'][idx], self.around_digit)), 
                    act=self.inv_action_idx[(np.around(_df['a1'][idx+1], self.around_digit), np.around(_df['a2'][idx+1], self.around_digit))],
                    next_obs=self._pos2state(np.around(_df['f1'][idx+1], self.around_digit), np.around(_df['f2'][idx+1], self.around_digit)),
                    rew=1.0))
            dataset.append(t)
        
            self.goal_states.append(self._pos2state(np.around(_df['f1'][_df.shape[0]-1], self.around_digit), 
                                                    np.around(_df['f2'][_df.shape[0]-1], self.around_digit)))

        return dataset
                
    def get_boundary(self, RMatrix, threshold=1.0):
        diff = np.zeros((RMatrix.shape[0], RMatrix.shape[1]))
        for i in range(1, RMatrix.shape[1]):
            diff[:, i] = RMatrix[:, i] - RMatrix[:, i-1]

        diff -= np.ones((RMatrix.shape[0], RMatrix.shape[1])) * threshold

        # print(diff)
        tmp_f2_idx, tmp_f1_idx = np.where(diff > 0)
        print(tmp_f2_idx)
        print(tmp_f1_idx)

        return np.min(tmp_f1_idx)

    @staticmethod
    def readFile(f_name):
        if os.path.exists(f_name):
            return pd.read_csv(f_name)
        else:
            logging.error('{0} does not exist!'.format(f_name))
            exit(1)

    @staticmethod
    def getDist(x1, y1, z1, x2, y2, z2):
        return np.sqrt((pow((x1-x2),2) + pow((y1-y2),2) + pow((z1-z2),2)))

    def extract_features(self, raw_data_path='/home/rdaneel/hri_costmap_traj/'):
        for _fin in os.listdir(raw_data_path):
            if _fin.split('.')[-1] == 'csv':
                file_name = raw_data_path + _fin
                traj_df = readFile(file_name)
                h_id = '1'
                r_id = '2'

                goal_pos2 = [traj_df['x'+r_id].tolist()[-1], traj_df['y'+r_id].tolist()[-1], traj_df['z'+r_id].tolist()[-1]]
                for idx, row in traj_df.iterrows():
                    traj_df.loc[idx, 'f1'] = np.around(getDist(row['x'+r_id], row['y'+r_id], row['z'+r_id], row['x'+h_id], row['y'+h_id], row['z'+h_id]), self.around_digit)
                    traj_df.loc[idx, 'f2'] = np.around(getDist(row['x'+r_id], row['y'+r_id], row['z'+r_id], goal_pos2[0], goal_pos2[1], goal_pos2[2]), self.around_digit)

                pre_row = pd.Series()
                for idx, row in traj_df.iterrows():
                    if idx == 0:
                        pre_row = row
                    else:
                        traj_df.loc[idx, 'a1'] = row['f1']-pre_row['f1']
                        traj_df.loc[idx, 'a2'] = row['f2']-pre_row['f2']
                        pre_row = row
                        
                traj_df.to_csv(raw_data_path + 'actions/' + _fin.split('.')[0] + '_action.csv')

    
    def main(self):
        np.random.seed(0)
        
        # phi
        phi = [self._feature_map(s) for s in range(self.observation_space().n)]
        phi = np.array(phi)

        print('Collect dataset ...')
        in_dataset = self.collect_demo()
        plot_dataset_distribution(in_dataset, (self.f1_length, self.f2_length), "Dataset State Distribution")
        print('Done!')

        # IRL
        print('Building IRL object ...')
        me_irl = MaxEntIRL(
            observation_space=self.observation_space(),
            action_space=self.action_space(),
            transition=self._construct_transition_fn(),
            goal_states=self.goal_states,
            dataset= in_dataset,
            feature_map=phi,
            max_iter=20,
            anneal_rate=0.9)
        print('Done!')

        print('\nStart training ...')
        Rprime = me_irl.train()
        Rprime = Rprime.reshape((self.f1_length, self.f2_length)).T

        print('Boundary index = ', self.get_boundary(Rprime))
        print('Radius = ', self.f1_idx[self.get_boundary(Rprime)])
        print(self.f1_idx)

        # plot results
        plot_grid_map(Rprime, "Reward (IRL)", print_values=True, cmap=plt.cm.Blues)
        plt.show()
        return
    
if __name__ == '__main__':
    costmap_irl = CostmapIRL()
    costmap_irl.main()