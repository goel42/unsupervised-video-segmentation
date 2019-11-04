import hdf5storage
import numpy as np
import os

def load_dataset(name, path):
    path = os.path.join(path, name+".mat")
    data = hdf5storage.loadmat(path)
    vid_features = [data['X'][i][0] for i in range(len(data['X']))]
    gt_labels = [data['Y'][i][0] for i in range(len(data['Y']))]
    vid_frames_count = [len(vid_features[i][0]) for i in range(len(vid_features))]
    gt_labels_count = [len(gt_labels[i][0]) for i in range(len(gt_labels))]

    assert len(vid_features) == len(gt_labels)
    assert vid_frames_count == gt_labels_count
    
#doubt: why subtract 1 for g_K? 
    g_K = len(data['labelC'][0])  -1 
    
#     print("Dataset Loaded Successfully")
 
    return vid_features, gt_labels, vid_frames_count, g_K

#test
# blah = ["cereals", "coffee", "friedegg", "juice", "milk", "pancake", "salat", "sandwich", "scrambledegg", "tea"]
# for item in blah:
#     _,_,_,_ = load_dataset(item, path)

#test
# path = r"/mnt/c/Users/dcsang/Documents/code_translate/DATA/breakfast_actions/formatted_data_new/coffee.mat"
# a,b, c, d = load_dataset("blah", path)

