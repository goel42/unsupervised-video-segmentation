def fs_subpart_embedding_theta_given(path2embeddingdata, embedding_labels, frames, subact_count, W):
    #TODO: verify var names for embedding_labels and frames
    
    #TODO: following always-true if condition for debugging
    #fillup condition from original code later
    if True:
        labels_vecs = fs_prep_embedding_data (path2embeddingdata, embedding_labels, frames, subact_count)
        W = defrag_run(path2embeddingdata, W)
        #TODO: code for saving
    else:
        #TODO: code for retrieving previously saved W and labels_vecs

        #TODO: for now omitting labels_vecs;finalising embedding portion
#     return W, labels_vecs
    return W
