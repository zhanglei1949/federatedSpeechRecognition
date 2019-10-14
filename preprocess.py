# transform audio to features and store them into two h5pys
# 1. Train test split ?
# 2. Client ids? 
#   shuffle into different groups and assign each group to a client
# 3. try a small dataset with 5 words first
# 4. ordereddict
# 5. feature extraction 

from constant import *
import h5py
import glob, collections
import numpy as np
import random
from utils.dataset_utils import extract_feature
def main():
    files = []
    labels = []
    for i,word in enumerate(WORDS):
        tmp = glob.glob(DATA_PATH + word + '/*.wav')
        files.extend(tmp)
        labels.extend([i]*len(tmp))
    print(len(files), np.unique(labels))

    #shuffle
    ids = np.arange(0, len(files)) 
    np.random.shuffle(ids)
    files = [files[i] for i in ids]
    labels = [labels[i] for i in ids]
    files = files[:100]
    labels = labels[:100]
    features = extract_feature(files)
    #expect a numpy array returned in shape (num_files, feature_dim_0, feature_dim_1)
    #(7000, 99, 161)
    #np.random.shuffle(files)
    labels = np.array(labels)
    features = np.array(features)
    print(labels.shape, features.shape)
    # write to h5py
    f = h5py.File(DATASET_FILENAME, 'w')
    group = f.create_group('examples')
    group_size = int(len(labels)/NUM_CLIENTS) 

    for i in range(0, len(labels), group_size):
        client_id = i
        sub = group.create_group(str(client_id))
        sub.create_dataset('label', data = labels[group_size*i : group_size*(i+1)])
        sub.create_dataset('pixels', data = features[group_size*i : group_size*(i+1)])
    f.close()
if __name__ == '__main__':
    main()
"""
    for i,f_ind in enumerate(ids):
        client_no = int(i/int(len(files)/NUM_CLIENTS))
        if client_no in dic.keys():
            dic[client_no].append(f_ind)
        else:
            dic[client_no] = [f]

    for k,v in dic.items():
        print(k,len(v))
"""
