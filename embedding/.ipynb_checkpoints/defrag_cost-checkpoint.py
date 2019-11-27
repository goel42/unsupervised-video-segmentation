#frames_batch should be BATCH_SIZE * frame_feats_size +1
#labels_batch should be BATCH_SIZE * word2vec_dim +1
#W_f should be latent_spze_dim * frame_feats_size+1
#W_a should be latent_space_dim * word2vec_dim +1

#TODO more verify that __batch_size == N == Ni == Ns
from utils.config_parse import params
import numpy as np
import math

def defrag_cost(frames_batch, labels_batch, W):
    __useglobal = True #TODO: most probably redundant since fs code does not use uselocal, if used put in config
    __gmargin = params["embedding"]["train"]["gmargin"] #TODO: better variable name [delta?] ? 
    __thr_globalscore = params["embedding"]["train"]["thrglobalscore"] #TODO: better variable name?
    __smoothnum = params["embedding"]["train"]["smoothnum"] #TODO: better variable name?
    __batch_size = params["embedding"]["train"]["batch_size"] #TODO: what if frames_batch created in defrag_run is less than batch_size
    
    W_f = W["Wf"]
    W_a = W["Wa"]
    
    gcost =0
    
    #dim should be latent_space_dim * BATCH_Size
    frames_latent_space = np.dot(W_f , frames_batch.T)
    #dim should be latent_space_dim * BATCH_Size
    labels_latent_space = np.dot(W_a , labels_batch.T) #remember label_batch has word2vecs of label strings
    
    dots = np.dot(frames_latent_space.T , labels_latent_space)
    
    #TODO: normalisation part insert here; not clear how done in orig code (SG,SGN)
    #DOUBT: gcost and cost in orig cost are same since cost is 0 initialised. removing gcost
    #DOUBT: I verified by running original codebase that SG == dots
    
    column_diffs = np.zeros([__batch_size, __batch_size])
    row_diffs = np.zeros([__batch_size, __batch_size])
    for i in range(__batch_size):
        curr_column_diff = np.maximum(0, dots[:, i] - dots[i,i] + __gmargin)
        curr_column_diff[i] = 0
        column_diffs[:,i] = curr_column_diff
        
        curr_row_diff = np.maximum(0,dots[i,:] - dots[i,i] + __gmargin)
        curr_row_diff[i] = 0
        row_diffs[i,:] = curr_row_diff
        
        gcost += sum(curr_column_diff) + sum(curr_row_diff)
    
    dsg = np.zeros([__batch_size, __batch_size])    
    for i in range(__batch_size):
        curr_column_diff = column_diffs[:,i]
        curr_row_diff = row_diffs[i,:]
        
        #column term backprop
        dsg[i,i] = dsg[i,i] - sum(curr_column_diff>0)
        dsg[:,i] = dsg[:,i] + (curr_column_diff>0)
        
        #row term backprop
        dsg[i,i] = dsg[i,i] - sum(curr_row_diff > 0)
        dsg[i,:] = dsg[i,:] - (curr_row_diff>0)
    
    ltop = np.zeros([__batch_size, __batch_size])
    ltopg = np.zeros(ltop.shape)
    for i in range(__batch_size):
        for j in range(__batch_size):
            d = dots[i,j]
            dd = dsg[i,j] #DOUBT: for now keeping it simple, since SGN matrix ==1, and d is a scalar
            if __thr_globalscore:
                if d<0 :
                    dd=0

            ltopg[i,j]=dd
    ltop = ltop + ltopg #DOUBT: redundant since ltop is 0. remove after clarifying with fadime
    
    #backprop intro fragment vectors
    #MAJOR #DOUBT alldeltas'img' == all'sent'vecs? why opposite ?
    #MAJOR #DOUBT why transpose of ltop in line 135 ie frames_deltas and not in labels_deltas?
    frames_deltas = np.dot(labels_latent_space, ltop.T) #MAJOR #TODO after clarifying
    labels_deltas = np.dot(frames_latent_space, ltop) #MAJOR #TODO after clarifying
    
    #backprop frame mapping
    df_W_f = np.dot(frames_deltas, frames_batch)
    
    #backprop label mapping
    df_W_a = np.dot(labels_deltas, labels_batch)
    
    #TODO:
#     BackwardSents() function not used to gen df_Wsem from original code. check with fadime?
    
    
    #TODO: verify iwth data 
    theta = np.concatenate([W_f.flatten(), W_a.flatten()])
    __regularisation_const = np.float(params["embedding"]["train"]["reg_const_gamma"])

    cost = {
        "raw_cost": gcost,
        "reg_cost": __regularisation_const/2 * np.sum(theta**2), #DOUBT: why divide by 2?
        "total_cost": None
    }
    cost["total_cost"] = cost["raw_cost"] + cost["reg_cost"]
    
    #TODO: verify iwth data 
    W_gradient = {
        "Wf_gradient": df_W_f + __regularisation_const*W_f,
        "Wa_gradient": df_W_a + __regularisation_const*W_a
    }
    
    return cost, W_gradient

    
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    