#frames_labels == samples.t
#frames_count == v_numbframes
#frames == X_probs;; i will take orginal frame features here and will cnvert to latent space ehre. x_probs already in latent space
#X_probs is something else; keeping same name for now change later
#params.mixGauss.no_gauss
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

import numpy as np
#TODO: value checking. pix_D in original code had 'inf' values when i was value checking. here in python i didnot come across any inf values. fishy?
#TODO: dim checking. done. looks fiine but checking by comparing values with matlab code TBD


#TODO: variable name:: whereever label is used specificy whether gauss_label or seg_label

def fit_gaussmix(X_probs_seg_label, gauss_labels):
    #TODO: following should instead be done by passing gauss_count as a paramter 
    unique_gauss_labels = np.unique(gauss_labels)
    gauss_count = len(unique_gauss_labels)
    
    gauss_mixtures = [] #contains gauss_count(3) num of guassian mixtures. each gaussian mixtures has 3 elements ie pi (scalar) , mu (size = subact_count) , sigma (subactcount*subactcunt)

    for idx in range(gauss_count):
        curr_gauss_label = unique_gauss_labels[idx]
        X_probs_seg_label_gauss_label = X_probs_seg_label[gauss_labels==curr_gauss_label, :]    #shape should be __ X subact_count
        
        pi = X_probs_seg_label_gauss_label.shape[0]/X_probs_seg_label.shape[0]
        mu = np.mean (X_probs_seg_label_gauss_label, axis =0) # TODO: size of this array should be "subact_count" 
        sigma = np.cov(X_probs_seg_label_gauss_label.T) + (0.0001 * np.eye(X_probs_seg_label.shape[1]))#should be subact_count x subact_count
        gauss_mixtures.append([pi,mu,sigma]) #TODO: verify dimension 
   
    return gauss_mixtures 

#gaussmixture_subact contains gauss_count(3) num of mixtures for 1 subact (not subacts)
#X_probs_seg_label contains X_probs for 1 subactivity
    #rgb_pts == X_probs_seg_label
def assign_gauss_single(X_probs_seg_label, gaussmixture_subact):
    gauss_count = len(gaussmixture_subact)
    #pix_D = np.zeros(np.sum(indices_curre_seg_label), gauss_count) #TODO basically, number of X-probs for a given seg_label X gauss_count
    pix_D = np.zeros([X_probs_seg_label.shape[0], gauss_count]) #TODO: verify dims

    for idx in range(gauss_count):
        pi_coeff = gaussmixture_subact[idx][0] #TODO: verify. should be scalar
        mu = gaussmixture_subact[idx][1]#TODO verify. dim -> subact_count
        sigma = gaussmixture_subact[idx][2]#TODO verify. dim -> subact_count X subact_count
        
        #TODO: verify col dim. should be  X_probs_seg_label.shape[0] X 1
        col = -1.0 * multivariate_normal.logpdf(X_probs_seg_label, mu, sigma) - np.log(pi_coeff) -1.5 * np.log(2*np.pi) #TODO: thoroughly check. "allow_singular" paramter default val used
        pix_D[:, idx] = col

    k_U = np.argmin(pix_D, axis =1) #should be  X_probs_seg_label.shape[0] X 1
    
    return pix_D, k_U

# def update_gauss_single(X_probs_seg_label, gauss_labels):
#     #UPDATE_GMM Part of GrabCut. Update the GMM parameters with newly assigned data 
#     #X_probs_curr_seg_label = X_probs[indices_curre_seg_label, :]
    
#     unique_gauss_labels = np.unique(gauss_labels)
#     gauss_count = len(unique_gauss_labels)

#     gauss_mixtures_updated = []

#     #TODO: the following is also used in fit_gmm; make a separate funcion and call it here
#     for idx in range(gauss_count):
#         curr_gauss_label = unique_gauss_labels[idx]
#         X_probs_seg_label_gauss_label = X_probs_seg_label[gauss_labels==curr_gauss_label, :]    #shape should be __ X subact_count

#         gauss_param_U[idx,0] = X_probs_seg_label_gauss_label.shape[0]/X_probs_seg_label.shape[0]
#         gauss_param_U[idx,1] = np.mean (X_probs_seg_label_gauss_label, axis =0) # size of this array should be "subact_count" 
#         gauss_param_U[idx,2] = np.cov(X_probs_seg_label_gauss_label.T) + (0.0001 * np.eye(X_probs_seg_label.shape[1]))#should be subact_count x subact_count
#     return gauss_param_U

def compute_unary( X_prob ,gaussmixtures_subact):
    
    pix_D, _ = assign_gauss_single(X_prob, gaussmixtures_subact)
    #TODO: does pixD really have inf vals? Matlab code does have inf vals 
    #TODO: is col (line 42) outputtng a scalar when we pass curr_val (dim of val -> subact_count)
    #line by line check assign function to see if it is giving expected resutls when we pass vector shape of 1X6 in X_probs_seg_label of asssign
    
    #TODO replace inf values in pix_D with 100
    
    D_min = np.min(pix_D) #TODO: basically need the minimum of three scalars in pix_D. do i need to specify axis?
    
    return D_min
    
def concat_label(vids_frames_label, vids_frames_count, vids_count):
    output =[]
    for vi in range(vids_count):
        fc = vids_frames_count[vi]
        temp = vids_frames_label[vi, :fc]
        output.extend(temp)
    return output

def concat(X_probs,vids_frames_count,vids_count,frames_count, subacts_count):
    output = np.zeros([subacts_count, frames_count])
    idx = 0
    for vi in range(vids_count):
        fc = vids_frames_count[vi]
        output[: ,idx:idx+fc] = X_probs[vi]
        idx += fc
    return output

#TODO: change curr_label to curr_subact
#TODO: using two diff variables for same thing: subacts_count and len(unique_subact_labels). to me their values will always be same? fix this
def compute_gauss_mix(vids_frames_label,vids_frames_count, X_probs, subacts_count):
    #num_clusters = #TODO from params
    num_clusters = 3
    vids_count = len(vids_frames_count)
    frames_count = sum(vids_frames_count)
    
    #TODO: change label names
    #remember X-probs[0] is 6 x 913 so need to transpose
    X_probs_concat = concat(X_probs,vids_frames_count,vids_count,frames_count, subacts_count)
    X_probs_concat = X_probs_concat.T #TODO: verify dims
    print("concat shape:", X_probs_concat.shape)
    
    frames_label_concat = concat_label(vids_frames_label, vids_frames_count, vids_count)
    frames_label_concat = np.array(frames_label_concat)
    
    unique_subact_labels = np.unique(frames_label_concat) #TODO: verify: should be 6
    
    
    gaussmixtures_subacts = [] #for each subact we have gauss_count number of gaussian mixtures. so 6 subacts and each subact has 3 mixs. 1 mix has 3 statistic props ie (_,mean,cov) 
    for i in range(len(unique_subact_labels)):
        curr_label = unique_subact_labels[i]
        indices_curr_label = frames_label_concat==curr_label
        #TODO: change name of following variable
        X_probs_frames_with_curr_label = X_probs_concat[indices_curr_label, :] 
        
        #Initialize GMM
        #TODO do we need the cityspaces distance metric? second parameter of 15 in original code is 10 for python default
        k_foreground = KMeans(n_clusters = num_clusters).fit(X_probs_frames_with_curr_label).labels_
        gaussmixure_curr_subact_label = fit_gaussmix(X_probs_frames_with_curr_label, k_foreground)
       
        #Assign GMM components 
        _ , k_foreground = assign_gauss_single(X_probs_frames_with_curr_label, gaussmixure_curr_subact_label)

        #Learn GMM parameters from data #UPDATE_GMM Part of GrabCut. Update the GMM parameters with newly assigned data 
        #in original code, the update_gmm_simgle is exactly the same as fit_gmm so reusing.
        gaussmixure_curr_subact_label = fit_gaussmix(X_probs_frames_with_curr_label, k_foreground)
        gaussmixtures_subacts.append(gaussmixure_curr_subact_label)
        
    #_probs_new = np.zeros([vids_count, 1])
    X_probs_new = []
    for vid_idx in range(vids_count):
        X_probs_curr =  X_probs[vid_idx]

        X_curr = np.zeros(X_probs[vid_idx].shape)
        for frame_idx in range(X_probs_curr.shape[1]): #TODO verify dim -> number of frames in current video
            curr_val = X_probs_curr[:, frame_idx] #TODO: is transpose needed?
            D_min = np.zeros(len(unique_subact_labels)) #TODO: verify #initialise with size because line 9 will give error
            for label_idx in range(len(unique_subact_labels)):
                curr_label = unique_subact_labels[label_idx]
                D_min[label_idx] = compute_unary( curr_val ,gaussmixtures_subacts[label_idx]) 
            X_curr[:, frame_idx] = D_min #TODO: or D_min.T

        X_probs_new.append(X_curr)

    return X_probs_new 






    
      
    

    