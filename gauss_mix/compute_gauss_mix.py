#frames_labels == samples.t
#frames_count == v_numbframes
#frames == X_probs;; i will take orginal frame features here and will cnvert to latent space ehre. x_probs already in latent space
#X_probs is something else; keeping same name for now change later
#params.mixGauss.no_gauss
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

import numpy as np

#TODO: variable name:: whereever label is used specificy whether gauss_label or seg_label

def fit_gmm(X_probs_seg_label, gauss_labels):
    unique_gauss_labels = np.unique(gauss_labels)
    gauss_count = len(unique_gauss_labels)
    gauss_param = np.zeros([gauss_count,3]) 

    for idx in range(gauss_count):
        curr_gauss_label = unique_gauss_labels[idx]
        X_probs_seg_label_gauss_label = X_probs_seg_label[gauss_labels==curr_gauss_label, :]    #shape should be __ X subact_count
        
        gauss_param[idx,0] = X_probs_seg_label_gauss_label.shape[0]/X_probs_seg_label.shape[0]
        gauss_param[idx,1] = np.mean (X_probs_seg_label_gauss_label, axis =0) # size of this array should be "subact_count" 
        gauss_param[idx,2] = np.cov(X_probs_seg_label_gauss_label.T) + (0.0001 * np.eye(X_probs_seg_label.shape[1]))#should be subact_count x subact_count

    return gauss_param

def assign_gauss_single(X_probs, indices_curre_seg_label, gauss_param_seg_label):
    gauss_count = gauss_param_seg_label.shape[0]
    pix_D = np.zeros(np.sum(indices_curre_seg_label), gauss_count) #TODO basically, number of X-probs for a given seg_label X gauss_count

    X_probs_curr_seg_label = X_probs[indices_curre_seg_label, :]

    for idx in range(gauss_count):
        pi_coeff = gauss_param_seg_label[idx, 0]
        mu = gauss_param_seg_label[idx, 1]
        sigma = gauss_param_seg_label[idx, 2]

        col = -1.0 * multivariate_normal.logpdf(X_probs_curr_seg_label, mu, sigma) -log(pi_coeff) -1.5*log(2*np.pi) #TODO: thoroughly check. allow_singular paramter default val used
        pix_D[:, idx] = col

    k_U = np.argmin(pix_D, axis =1) #should be np.sum(indices_curre_seg_label) X 1

def update_gauss_single(X_probs_seg_label, indices_curre_seg_label, gauss_labels):
    #UPDATE_GMM Part of GrabCut. Update the GMM parameters with newly assigned data 
    X_probs_curr_seg_label = X_probs[indices_curre_seg_label, :]
    
    unique_gauss_labels = np.unique(gauss_labels)
    gauss_count = len(unique_gauss_labels)

    gauss_param_U = np.zeros(gauss_count, 3)

    #TODO: the following is also used in fit_gmm; make a separate funcion and call it here
    for idx in range(gauss_count):
        curr_gauss_label = unique_gauss_labels[idx]
        X_probs_seg_label_gauss_label = X_probs_seg_label[gauss_labels==curr_gauss_label, :]    #shape should be __ X subact_count

        gauss_param_U[idx,0] = X_probs_seg_label_gauss_label.shape[0]/X_probs_seg_label.shape[0]
        gauss_param_U[idx,1] = np.mean (X_probs_seg_label_gauss_label, axis =0) # size of this array should be "subact_count" 
        gauss_param_U[idx,2] = np.cov(X_probs_seg_label_gauss_label.T) + (0.0001 * np.eye(X_probs_seg_label.shape[1]))#should be subact_count x subact_count
    return gauss_param_U

def compute_unary
    #TODO: is very similar to assign_gauss_single

def compute_gauss_mix(frames_label,vids_frames_count, X_probs):
    
    num_clusters = #TODO from params
    vids_count = len(vids_frames_count)
    frames_labels_without_padding = #TODO
    
    #TODO: change label names
    X_probs_concat = #TODO [must be frames_totalcount X subact_count e.g. 97290 X 6]
    frames_label_concat = #TODO [must be frames_total_count X 1 e.g. 97290 X 1]
    
    unique_seg_labels = #TODO 
    
    for i in range(unique_labels):
        curr_label = unique_labels[i]
        indices_curr_label = frames_label_concat==curr_label
        #TODO: change name of following variable
        X_probs_frames_with_curr_label = #TODO => should be X_probs_concat[indices_curr_label, :]; 
        
        #Initialize GMM
        #TODO do we need the cityspaces distance metric? second parameter of 15 in original code is 10 for python default
        k_foreground = KMeans(n_clusters = num_clusters).fit(X_probs_frames_with_curr_label).labels_
        gauss_mix_foreground[i] = fit_gmm(X_probs_frames_with_curr_label, k_foreground)

        #Assign GMM components 
        k_foreground = #implement fn

        #Learn GMM parameters from data
        gauss_mix_foreground[iop] = #TODO implement fn update_gauss_mix_single()

    X_probs_new = np.zeros([vids_count, 1])

    for vid_idx in range(vids_count):
        X_probs_curr =  X_probs[vid_idx]

        X_curr = [] #TODO: initialise
        for frame_idx in range(): #TODO number of frames in current video
            curr_val = X_probs_frames_with_curr_label[:, frame_idx] #TODO: is transpose needed?
            D_min = [] #initialise with size because line 9 will give error
            for label_idx in range(len(unique_labels)):
                curr_label = unique_labels[label_idx]
                D_min[label_idx] = #TODO compute_unary(s) #this line should give error since D_min size not initialised
            X_curr[:, frame_idx] = D_min #TODO: or D_min.T

        X_probs_new[vid_idx] = X_curr

    return X_probs_new 






    
      
    

    