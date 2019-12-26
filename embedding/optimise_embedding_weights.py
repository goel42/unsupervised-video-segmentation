from embedding.defrag_run import defrag_run
import numpy as np


#vids_frames_label could either be ground truth labels or the labels estimated by GMM or random
#TODO what is metadata, stoi and itos in the original code. is it used anywhere? skipped for now
#TODO write code for saving train test val splits to file
def prep_data(path, vids_frames_features, vids_frames_count, vids_frames_label, subacts_count, labels_features):
    #TODO: take from params, skipping for now
    #pval = #TODO
    pval = 0
    
    #frames_count = sum(vids_frames_count) #TODO verify output
    frames_all = []
    labels_all = []
    
    for frames_features, frames_label in zip(vids_frames_features, vids_frames_label):
        for frame_idx in range(frames_features.shape[1]):
            #frames_label contains 0 padding at the end (if it is coming from random initialisation) ie why using frame_idx to iterate 
            #that way, we will not be accessing the 0 padding at the end
            frame_features = frames_features[:, frame_idx]
            #DOUBT: why make dim from 64 to 65? #TODO: append not memory/time efficient
            frame_features = np.append(frame_features,1)
            
            label = frames_label[frame_idx]
            label_features = labels_features[label]["vec"] #TODO verify
            label_features = np.append(label_features,1)
            
            frames_all.append(frame_features)
            labels_all.append(label_features)
    
    frames_count = len(frames_all)
    assert frames_count == sum(vids_frames_count)
    
    rp = np.random.permutation(frames_count)
    validation_size = int(np.floor((frames_count*pval)/100))
    train_size = int(frames_count - validation_size)
    
    train_indices = sorted(rp[0:train_size]) #TODO verify
    validation_indices = sorted(rp[train_size: ]) #TODO verify
    
    frames_all = np.array(frames_all)
    labels_all = np.array(labels_all)
    
    train_set = {"frames": frames_all[train_indices],
                 "labels_vecs": labels_all[train_indices]} #TODO verify if it contains same label for each frame
    
    validation_set = {"frames": frames_all[validation_indices],
                      "labels_vecs": frames_all[validation_indices]}

    #TODO save train and validation set to file
    return train_set, validation_set
    
    

def optimise_embedding_weights(path, vids_frames_features,
                               vids_frames_count, subacts_count, vids_frames_label, W, label_features):
    
    print("Started optimising W")
    train, vald = prep_data(path, vids_frames_features, vids_frames_count, vids_frames_label, subacts_count, label_features)
    W = defrag_run(path, train, vald, W)
    return W