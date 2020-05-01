import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def _pos2state(x, y, z, x_length, y_length):
    print ("_pos2state: ", x, y, z)

    tmp_x = int(np.around(x, 1) * 10)
    tmp_y = int(np.around(y, 1) * 10)
    tmp_z = int(np.around(z, 1) * 10)

    tmp_x_length = int(np.around(x_length, 1) * 10)
    tmp_y_length = int(np.around(y_length, 1) * 10)

    # print ((z * 10 * x_length * 10 * y_length * 10) + (y * 10 * x_length * 10) + x * 10)
    return (tmp_z * tmp_x_length * tmp_y_length) + (tmp_y * tmp_x_length) + tmp_x

def _state2pos(idx, x_length, y_length):
    z = int(idx // (x_length * y_length));
    idx -= (z * x_length * y_length);
    y = int(idx // x_length);
    x = int(idx % x_length);
    print ('(x, y, z) = ({0}, {1}, {2})'.format(z, y, x))
    return (x, y, z);

if __name__ == '__main__':
    csv_path = '/home/rdaneel/hri_costmap_traj/actions/'
    dp_list = list()
    file_name = list()
    goal_pos2 = list()

    for _fin in os.listdir(csv_path):
        dp_list.append(pd.read_csv(csv_path + _fin))
        file_name.append(_fin)
    
    for _dp in dp_list:
        # get position of last row
        goal_pos1 = [_dp['x1'].tolist()[-1], _dp['y1'].tolist()[-1], _dp['z1'].tolist()[-1]]
        goal_pos2 = [_dp['x2'].tolist()[-1], _dp['y2'].tolist()[-1], _dp['z2'].tolist()[-1]]

        # position relative to the goal
        _dp['x1'] -= goal_pos2[0]
        _dp['y1'] -= goal_pos2[1]
        _dp['z1'] -= goal_pos2[2]

        _dp['x2'] -= goal_pos2[0]
        _dp['y2'] -= goal_pos2[1]
        _dp['z2'] -= goal_pos2[2]

    # Get the maximum position among all demos
    tmp_x = list()
    tmp_y = list()
    tmp_z = list()
    for _dp in dp_list:
        tmp_x.append(np.around(max(max(_dp['x1'].tolist()), max(_dp['x2'].tolist())), 1))
        tmp_y.append(np.around(max(max(_dp['y1'].tolist()), max(_dp['y2'].tolist())), 1))
        tmp_z.append(np.around(max(max(_dp['z1'].tolist()), max(_dp['z2'].tolist())), 1))

    max_x = max(tmp_x)
    max_y = max(tmp_y)
    max_z = max(tmp_z)

    # Get the minimum position among all demos
    tmp_x = list()
    tmp_y = list()
    tmp_z = list()
    for _dp in dp_list:
        tmp_x.append(np.around(min(min(_dp['x1'].tolist()), min(_dp['x2'].tolist())), 1))
        tmp_y.append(np.around(min(min(_dp['y1'].tolist()), min(_dp['y2'].tolist())), 1))
        tmp_z.append(np.around(min(min(_dp['z1'].tolist()), min(_dp['z2'].tolist())), 1))

    print(file_name)
    print(tmp_x)
    print(tmp_y)
    print(tmp_z)

    min_x = min(tmp_x)
    min_y = min(tmp_y)
    min_z = min(tmp_z)

    print('max: ({0}, {1}, {2})'.format(max_x, max_y, max_z))
    print('min: ({0}, {1}, {2})'.format(min_x, min_y, min_z))

    x_length = np.around(max_x - min_x, 1)
    y_length = np.around(max_y - min_y, 1)
    z_length = np.around(max_z - min_z, 1)

    print (x_length, y_length, z_length)

    for _dp in dp_list:
        for idx, row in _dp.iterrows():
            human_state = _pos2state(row['x1'], row['y1'], row['z1'], x_length, y_length)
            robot_state = _pos2state(row['x2'], row['y2'], row['z2'], x_length, y_length)
            print('human, robot = {0}, {1}'.format(human_state, robot_state))
