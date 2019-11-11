import numpy as np
from vToPi import vToPi as v_to_pi

# omitting samples.s in original file, looks redundant
#vids_frames_count == numSents
#subactivity_count = K
#rho_prior = rhoprior
#vids_count == numDocs
#frames_max_count == maxNumSents

#TODO to verify the dimensions of all the following data structures and their counts

def initialize_subactivity_assignments(vids_frames_count, subactivity_count, rho_prior):
    vids_count = len(vids_frames_count)
    frames_max_count = max(vids_frames_count)
    
    keep =1
    
    #DOUBT: how are we using this circular queue?
    #Tail of circular queue
    tail =1
    
    #counting all subactivities including SIL
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
        activity_counts = np.sort(np.ceil(np.random.uniform(0, subactivity_count, (1, vids_frames_count[i])))).astype(np.uint32)
#         print(activity_counts[0])
        pi = v_to_pi(samples[tail-1]['v'][:, i])
#         print(samples[tail-1]['v'][:, i])
#         print(samples[tail-1]['v'][:, i].shape)
        print("###")
        print(pi)
#         print(pi.shape)
        print("%%%")
#         pi = v_to_pi([0,0,0,0])
        for j in range(vids_frames_count[i]):
            samples[tail-1]['z'][i][j] = np.uint32(pi[activity_counts[0][j]-1])

            #INCOMPLETE
            #test the dimensions and indexing of all these data structures
            #compare outputs from matlab code and python code
            #implement and verify vToPi function. done.
    print("Vid over")
    return samples
s = initialize_subactivity_assignments([10,30,40,50], 5,1 )
print (s)
    
    
    
    
    