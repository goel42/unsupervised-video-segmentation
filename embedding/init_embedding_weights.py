#TODO: Run, Debug and clarify #DOUBT

#init_embedding_weights == init_params
from utils.config_parse import params
import numpy as np
import math


def get_matrix(fan_in, fan_out):
    __range = 1/math.sqrt(6*fan_in + fan_out) #DOUBT: how this range calculated?
    W_temp = -(__range) + (2*__range)*np.random.rand(fan_out, fan_in) #TODO: second term has to be elementwise multiplication. to test.
    return W_temp
    

#initialising the transformation matrices for the latent space of the embeddings
def init_embedding_weights():

    __fan_out = params["embedding"]["train"]["latent_space_dim"]
    
    #for labels
    __fan_in = params["embedding"]["train"]["label_vec_dim"] + 1 #DOUBT: +1 for bias? 1) how bias? why +1?
    W_a = get_matrix(__fan_in, __fan_out)
    
    #for frames
    __fan_in = params["embedding"]["train"]["frame_features_dim"] +1 #DOUBT: refer my evernote notes 
    W_f = get_matrix(__fan_in, __fan_out)
    
    W = {"Wa": W_a, "Wf": W_f}
    return W
    

    
