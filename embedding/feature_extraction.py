# -*- coding: utf-8 -*-
#TODO: better name for function
import numpy as np
def forward_sent(label_features, Wa):
    vec = label_features["vec"]
    vec= np.append(vec,1)
    
    vec_latent_space = np.dot(Wa, vec) #TODO: why not vec.T here like in line 24 for frame_features_latent
    #TODO: in original code(ForwardSent), activation function is mentioned but not used. put it here if needed
    return vec_latent_space

def extract_features(labels_features, vids_frames_features, W):
    
    
    #TODO: take the following two __vars from params
    __thrglobalscore = 0
    __keep_neg_response =  0
    __smoothnum = 0
    
    
    #labels_count == subacts_count elsewhere.
    Wa = W["Wa"]
    Wf = W["Wf"]
    labels_count = len(labels_features)
    
    X_probs = []
    
    #TODO: verify opt
    labels_features_latent = [ forward_sent(label_features, Wa) for label, label_features in labels_features.items() ]
    
    for vid_idx, frames_features in enumerate(vids_frames_features):
        frames_probs = np.zeros([labels_count, frames_features.shape[1] ])
        for frame_idx, frame_features in enumerate(frames_features.T):
            frame_features = np.append(frame_features, 1) 
            frame_features_latent = np.dot(Wf, frame_features) #TODO verify dot product
            
            feat_resp = np.zeros(labels_count)
            for idx, label_features_latent in enumerate (labels_features_latent):
                d = np.dot(frame_features_latent.T , label_features_latent) #TODO verify dot product
            
                if __thrglobalscore:
                    d = max(0,d)
                #TODO: in original code, sum etc is calculated since d is 1 element, skippin git
                feat_resp[idx]= d
            
            if __keep_neg_response:
                feat_resp[feat_resp<0] = 0
            frames_probs[:, frame_idx] = feat_resp
                
        X_probs.append(frames_probs)
    return X_probs
            
            
        
    
    
    
