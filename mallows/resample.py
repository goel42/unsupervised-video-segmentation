from mallows.utility_mallows import sample_normalised
import numpy as np
from utils.vToPi import vToPi
#frames_subacts_label == vids_frames_label in remaining portion of code

def resample_z(vid_idx, vids_frames_subacts_label, vids_frames_invcount, vids_frames_mixgauss, subacts_count, vids_frames_count, ntau, ntaus, tau_0):
    print("Entered resample_z")
    order = np.random.permutation(vids_frames_count[vid_idx]) #randomly generated seq to process frames
    
    #getting X_mixgauss for current video vid_idx
    frames_mixgauss = vids_frames_mixgauss[vid_idx]
    frames_count = vids_frames_count[vid_idx]
    
    subacts_order = vToPi(vids_frames_invcount[:,vid_idx])
    
    for i in range(frames_count):
        curr_frame_idx = round(order[i])
        subact_curr = vids_frames_subacts_label[vid_idx][curr_frame_idx] #TODO: verify
        
        ntau[subact_curr-1] -= 1 # subacts_curr "-1" because subacts are 1-indexed
        ntaus -= 1
        
        subacts_prob = np.zeros(subacts_count)
        for sa in range(1, subacts_count+1):
            proposed_subacts = vids_frames_subacts_label[vid_idx] #this contains frames_max_count items whereas i just need equivalent to current num of frames. do i need to change nayhting?
            proposed_subacts[curr_frame_idx] = sa
            #proposed_subacts[:frames_count].sort(key = lambda x: np.where(subacts_order==x)) #TODO take care that subacts here are 1-indexed and in C code it is 0-indexed #MAJOR DOUBT
            proposed_subacts[:frames_count] = sorted(proposed_subacts[:frames_count], key = lambda x: np.where(subacts_order==x) )
            
            result = 0
            for f in range(frames_count):
                # -1 because subacts are 1-indexed
                result = result + frames_mixgauss[proposed_subacts[f]-1, f] #TODO verify the params to frames_mixgauss 
            
            subacts_prob[sa-1] = np.log((float(ntau[sa-1]) + tau_0)/(float(ntaus) + (tau_0 * float(subacts_count)))) - result       
        
        #+1 in subacts_curr because 1-indexed
        subact_curr = sample_normalised(subacts_prob, subacts_count) + 1 #TODO see comment in sample_normalised line 6 (_rand)
        vids_frames_subacts_label[vid_idx][curr_frame_idx] = subact_curr
        
        ntau[subact_curr-1] += 1
        ntaus +=1
        
    return vids_frames_subacts_label, ntau, ntaus

#TODO: the subactivities are semantically in range [1,K] but when used as index -1 needs to be done. thoroughly check.
def resample_v(vid_idx,vids_frames_subacts_label, vids_frames_invcount,vids_frames_mixgauss, subacts_count, vids_frames_count, nvs, rho):
    print("Entered resample_v")
    order = np.random.permutation(subacts_count-1) #randomly gen seq in which we process each element of inversion_count

    #getting X_mixgauss for current video vid_idx
    frames_mixgauss = vids_frames_mixgauss[vid_idx]
    frames_count = vids_frames_count[vid_idx]
    
    proposed_v = vids_frames_invcount[:,vid_idx]
    proposed_subacts = vids_frames_subacts_label[vid_idx]
    
    for v_idx in order:
        nvs[v_idx] -= vids_frames_invcount[v_idx, vid_idx] #TODO can proposed_v be used here?
        
        curr_v_prob = np.zeros(subacts_count - v_idx)
        for k in range(subacts_count - v_idx): #TODO: copy my comments from C code here
            proposed_v[v_idx] = k
            subacts_order = vToPi(proposed_v)
            #proposed_subacts[:frames_count].sort(key = lambda x:np.where(subacts_order==x))
            proposed_subacts[:frames_count] = sorted(proposed_subacts[:frames_count],key = lambda x:np.where(subacts_order==x) )
            
            result =0
            for f in range(frames_count):
                result = result + frames_mixgauss[ proposed_subacts[f]-1, f]
            curr_v_prob[k] = -result - (rho[v_idx]*float(k)) 
            
        curr_v = sample_normalised(curr_v_prob, subacts_count - v_idx)
        vids_frames_invcount[v_idx, vid_idx] = curr_v
        proposed_v[v_idx] = curr_v
        nvs[v_idx] += curr_v

    return vids_frames_invcount, nvs #TODO: return rho? any returnable whose value is changed and not added here?
    #return g_v ,g_nvs, g_rho 
    #g_nvs is used in sample_rho, 

#TODO   
# def resample_rho():

# 	rho_Array = []

# 	for i in range(g_K-1):
# 		#TODO
