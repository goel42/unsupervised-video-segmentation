from mallows.sample import sample_normalised

#frames_subacts_label == vids_frames_label in remaining portion of code
#MAJOT TODO: frames_subacts_order is to be used for comparator of sort (unfinished). not sure whether this is actual subact_order or the inversion_count(v)
#MAJOR TODO: the samples.z passed here will have frames_max_count number of frames. now what changes?

def resample_z(vid_idx, vids_frames_subacts_label, frames_subacts_order, vids_frames_mixgauss, ntau, ntaus, subacts_count, vids_frames_count):

    order = np.ramdom.permutation(vids_frames_count[vididx]) #randomly generated seq to process frames
    
    #getting X_mixgauss for current video vid_idx
    frames_mixgauss = vids_frames_mixgauss[vid_idx]
	frames_count = vids_frames_count[vid_idx]
    
    #TODO: setglobalordering wali shit here..
    
    for i in range(frames_count):
        curr_frame_idx = round(order[i])
        subact_curr = vids_frames_subacts_label[vid_idx][curr_frame_idx] #TODO: verify
        
        ntau[subact_curr-1] -= 1 #-1 because subacts are 1-indexed
        ntaus -= 1
        
        subacts_prob = np.zeros(subacts_count)
        for sa in range(1, subacts_count+1):
            proposed_subacts = vids_frames_subacts_label[vid_idx] #this contains frames_max_count items whereas i just need equivalent to current num of frames. do i need to change nayhting?
            proposed_subacts[curr_frame_idx] = sa
            sort( ) #TODO take care that subacts here are 1-indexed and in C code it is 0-indexed #MAJOR DOUBT
            
            result = 0
            for f in range(frames_count):
                result = result + frames_mixgauss[] #TODO 
            
            subacts_prob[sa-1] = np.log((float(ntau[sa-1]) + ntau_0)/(float(ntaus) + (ntau_0 * float(subacts_count)))) - result       
        
        subact_curr = sample_normalised() #TODO
        vids_frames_label[vid_idx][curr_frame_idx] = subact_curr
        
        ntau[subact_curr-1] += 1
        ntaus +=1
        
    return vids_frames_label, ntau, ntaus

#TODO: the subactivities are semantically in range [1,K] but when used as index -1 needs to be done. thoroughly check.
def resample_v(rho):
    order = np.ramdom.permutation(subacts_count-1) #randomly gen seq in which we process each element of inversion_count

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
            #TODO sort and setglobalpermutation wali bkc
            
            result =0
            for f in range(frames_count):
                result = result + frames_mixgauss[ proposed_subacts[f]-1, f]
            curr_v_prob[k] = -result - (rho[v_idx]*float(k)) 
            
         curr_v = sample_normalised(curr_v_prob, subacts_count - v_idx, /*todo*/ )
         vids_frames_invcount[v_idx, vid_idx] = curr_v
         porposed_v[v_idx] = curr_v
         nvs[v_idx] += curr_v

    return g_nvs, g_v , g_rho #TODO fill in more returnables
    #g_nvs is used in sample_rho, 
    
def resample_rho():

	rho_Array = []

	for i in range(g_K-1):
		#TODO
