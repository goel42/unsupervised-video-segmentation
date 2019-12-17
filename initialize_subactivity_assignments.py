import numpy as np
from utils.vToPi import vToPi as v_to_pi

# omitting samples.s in original file, looks redundant
#vids_frames_count == numSents
#subactivity_count = K
#rho_prior = rhoprior
#vids_count == numDocs
#frames_max_count == maxNumSents

#TODO to verify the dimensions of all the following data structures and their counts
#DOUBT: v is 0 so pi always be 1,2,3,4...

def initialize_subactivity_assignments(vids_frames_count, subactivity_count, rho_prior):
    vids_count = len(vids_frames_count)
    frames_max_count = max(vids_frames_count)
    
    keep =1
    
    #DOUBT: how are we using this circular queue?
    #Tail of circular queue
    tail =1
    
    #DOUBT #TODO counting all subactivities including SIL (check from load_dataset.py)
    sample = {'z': np.zeros((vids_count, frames_max_count), np.uint32),
              # -1 in v because it is inversion count
              'v': np.zeros((subactivity_count-1, vids_count), np.uint32),
              #DOUBT: check if count of rho items is K or K-1; following original code for now
              'rho': rho_prior * np.ones((subactivity_count-1,1))
             }
    samples = [sample]
    
    #TODO: skipped the drawing based on probability, not clear in code. So, samples[0].v is initialised with 0
    for i in range(vids_count):
        #uniform distribution within [low, high)
        #TODO:  also try np.floor so that python's 0-based indexing is followed automatically
        #DOUBT: an activity coud be labelled as 0 and ceil would also be 0. VERY UNLIKELY CHNACE but possible. correct it
        activity_counts = np.sort(np.ceil(np.random.uniform(0, subactivity_count, (1, vids_frames_count[i])))).astype(np.uint32)
        pi = v_to_pi(samples[tail-1]['v'][:, i])
        for j in range(vids_frames_count[i]):
            samples[tail-1]['z'][i][j] = np.uint32(pi[activity_counts[0][j]-1])

    print("Uniform initialisation of subactivity labels complete")
    return samples








# s = initialize_subactivity_assignments([10,30,40,50], 5,1 )
# print (s)
    
    
    
    
    