def forward_sent(subact, W_a):
    N = len(subact.label_str) #TODO should give the length of the string. verify
    Z = np.zeros(params.embedding.train.semantic_space, 1) #TODO why not +1 here?
    v_cat = np.zeros(params.embedding.train.word_dim, 1) #TODO why not +1 here, bias thingy?
    
#     for i in range(N):
        #TODO
        #MAJOR #DOUBT the original matlab code implies that we are adding label vector to itself N times?
        #MAJOR #DOUBT since N is length of label string which is usually 1 (single digit labels) why this for loop when it will exec only once?
    
    v_cat = np.append(v_cat,1)
    v_lin = np.dot(W_a,v_cat)
    #DOUBT #TODO 'v' is unused variable in original code. skipping it
    Z[:, 0] = v_lin
    return Z

def defrag_eval(split, W):
    __thr_globalscore = params.embedding.train.thrglobalscore #TODO: better variable name?
    __smoothnum = params.embedding.train.smoothnum #TODO: better variable name?
    
    W_f = W["Wf"]
    W_a = W["Wa"]
    
    frames_count = #TODO
    subacts_count = #TODO
    
    frames_batch = #TODO
    subacts = #TODO split.textfeat
    
    #direct matrix multiplication yields us the same result in one go for entire batch without needing a for loop
    frames_latent_space = np.dot(W_f , frames_batch.T)
    
    #
    M = np.zeros(subacts_count, frames_count)
    for i in range(subacts_count):
        z = forward_sent(subacts[i], W_a)
        if z: #TODO #should work, but verify #checking if z is empty
            M[i,:] = -np.inf
            continue
        for j in range(frames_count):
            d = np.dot(frames_latent_space[:,j].T , z)
            if __thr_globalscore:
                d = max(0,d)
            #DOUBT #TODO do we need to normalise d? original code logic of normalisation is invalid here
            M[i,j] = d
            
    E2_ranks = np.zeros(subacts_count, 1);
    for s in range(subacts_count):
        ind = M[s, :].argsort()[::-1] #TODO reverse to get descending. check and verify with data
        curr_frame_id = #TODO stoi thingy
        idxs = #TODO complicated matlab inline function
        #TODO beware that these are 0-based indices. match and verify with matlab's 1-based
        rank_of_closest = min(idxs)
        E2_ranks(s) = rank_of_closest
    
    frames_id = #TODO itos thingy
    M2 = M.T
    E3_ranks = np.zeros(M2.shape[0],1)
    for i in range(M2.shape[0]):
        ind = M2[i, :].argsort()[::-1]
        curr_frame_id = frames_id[i]
        rank_of_closest = #TODO
        E3_ranks[i] = rank_of_closest
        
    return E2_ranks, E3_ranks
    
            
 