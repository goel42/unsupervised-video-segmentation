import hdf5storage
import numpy as np
import os


def get_path(dir_path, fold_type, data_type):
    return os.path.join(dir_path, "split_"+fold_type+"_"+data_type+".mat")
    
def get_dataset_fold(fold_type, path2data_dir):
    """fold_type: val, train or test"""
    
    meta_path = get_path(path2data_dir, fold_type, "meta")
    labels_path = get_path(path2data_dir, fold_type, "sent")
    frames_path = get_path(path2data_dir, fold_type, "img")
    
    metadata = hdf5storage.loadmat(meta_path)
    labeldata = hdf5storage.loadmat(labels_path)
    framesdata = hdf5storage.loadmat(frames_path)
    
    frames =[]
    labels_appended = []
    if framesdata["Img"].size > 0:
        #for the time being; 
        frames = np.array([framesdata["Img"][i][0][0][0][0][0] for i in range(framesdata["Img"].shape[0])])
        labels = np.array([labeldata["Text"][0][i][0][0][2].T[0] for i in range(labeldata["Text"].shape[1])])

        #making word_2_vec dimension =201
        concat = [1]
        labels_appended = np.zeros( (labels.shape[0], labels.shape[1]+1))
        for i in range(labels.shape[0]):
            labels_appended[i] = np.concatenate((labels[i],concat))
    else:
        print("no frames")
    
    data = {
        "meta": metadata["meta"],
        "itos": metadata["itos"],
        "stoi": metadata["stoi"],
        "TextFeat": labeldata["TextFeat"],
        "frames_orig": labeldata["Text"],
        "labels_vecs": labels_appended,
        "frames_orig": framesdata["Img"],
        "frames": frames
    }

    return data

# def get_path(dir_path, fold_type, data_type):
#     return os.path.join(dir_path, "split_"+fold_type+"_"+data_type+".mat")
    
# def get_dataset_fold(fold_type, path2data_dir):
#     """fold_type: val, train or test"""
    
#     meta_path = get_path(path2data_dir, fold_type, "meta")
#     labels_path = get_path(path2data_dir, fold_type, "sent")
#     frames_path = get_path(path2data_dir, fold_type, "img")
    
#     metadata = hdf5storage.loadmat(meta_path)
#     labeldata = hdf5storage.loadmat(labels_path)
#     framesdata = hdf5storage.loadmat(frames_path)
    
#     frames_features = [framesdata["Img"][i][0][0][0][0][0] for i in range(framesdata["Img"].shape[0])]
#     label = np.array([blah["frames_labels_vecs"][0][i][0][0][2].T[0] for i in range(blah["frames_labels_vecs"].shape[1])])
#     data = {
#         "meta": metadata["meta"],
#         "itos": metadata["itos"],
#         "stoi": metadata["stoi"],
#         "TextFeat": labeldata["TextFeat"],
#         "frames_labels_vecs": labeldata["Text"],
#         "frames_features": framesdata["Img"]
#     }
    
#     return data

    