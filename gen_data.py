# coding:utf-8

import numpy as np 

# convert CharacterTrajectory into Acceleration

def gen_acceleration():
    datapath = "/home/scw4750/songbinxu/datasets/CharacterTrajectories/CharacterTrajectories.npz"

    trajectory = np.load(datapath)

    acceleration = {}
    acceleration['y_train'] = trajectory['y_train']
    acceleration['y_test'] = trajectory['y_test']

    acceleration['x_train'] = np.asarray([np.diff(np.diff(seq.T)).T for seq in trajectory['x_train']])
    acceleration['x_test'] = np.asarray([np.diff(np.diff(seq.T)).T for seq in trajectory['x_test']])

    np.savez("/home/scw4750/songbinxu/datasets/CharacterTrajectories/Acceleration.npz", **acceleration)

# gen_acceleration()
    
def get_lower():
    not_used = [ord(x) - ord('a') for x in ['f','i','j','k','t','x']]
    used = []
    for i in range(26):
        if not i in not_used:
            used.append(i)
            
    data_path = "/home/scw4750/songbinxu/autoencoder/data/lower.npz"

    lower = np.load(data_path)
    # f i j k t x
    # 'a' 'b' 'c' 'd' 'e' 'g' 'h' 'l' 'm' 'n' 'o' 'p' 'q' 'r' 's' 'u' 'v' 'w' 'y' 'z' 

    indexes = []
    for i in range(len(lower['XYZ'])):
        if not np.argmax(lower['labels'][i]) in not_used:
            indexes.append(i)
    
    lower_20 = {}
    for k in lower.keys():
        lower_20[k] = lower[k][indexes]
    lower_20['labels'] = lower_20['labels'][:, used]
    print lower_20['labels'].shape
    np.savez("/home/scw4750/songbinxu/autoencoder/data/lower_20.npz", **lower_20)


get_lower()



    