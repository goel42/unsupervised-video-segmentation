#frames_labels == samples.t
#frames_count == v_numbframes
#frames == X_probs;; i will take orginal frame features here and will cnvert to latent space ehre. x_probs already in latent space
#X_probs is something else; keeping same name for now change later
#params.mixGauss.no_gauss

def compute_gauss_mix(frames_label,vids_frames_count, X_probs):
    
    vids_count = len(vids_frames_count)
    frames_labels_without_padding = #TODO
    
    #TODO: change label names
    X_probs_concat = #TODO [must be frames_totalcount X subact_count e.g. 97290 X 6]
    frames_label_concat = #TODO [must be frames_total_count X 1 e.g. 97290 X 1]
    
    unique_labels = #TODO 
    
    for i in range(unique_labels):
        curr_label = unique_labels[i]
        indices_curr_label = frames_label_concat==curr_label
        #TODO: change name of following variable
        X_probs_frames_with_curr_label = #TODO => should be X_probs_concat[indices_curr_label, :]; 
        
        #Initialize GMM
        
    
      
    

    