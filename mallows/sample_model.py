# -*- coding: utf-8 -*-
from mallows.utility_mallows import v_prior
import numpy as np
from utils.vToPi import vToPi
from mallows.resample import resample_z, resample_v

def sample_model(samples, vids_frames_mixgauss,subacts_count, vids_count,vids_frames_count, ntau,ntaus, tau_0, nvs, iterations):
    print("Entered Sampling of Mallows")
    z = samples[0]["z"]
    v = samples[0]["v"]
    rho = samples[0]["rho"]
    
    for vid_idx in range(vids_count):
        subacts_order = vToPi(v[:,vid_idx])
        # z[vid_idx, :vids_frames_count[vid_idx]].sort(key = lambda x:np.where(subacts_order==x))
        z[vid_idx, :vids_frames_count[vid_idx]] = sorted(z[vid_idx, :vids_frames_count[vid_idx]], key = lambda x:np.where(subacts_order==x))

    for iters in range(iterations):
        for vid_idx in range(vids_count):
            
            z, ntau,ntaus = resample_z(vid_idx, z, v, vids_frames_mixgauss, subacts_count, vids_frames_count, ntau, ntaus, tau_0)
            
            v, nvs = resample_v(vid_idx, z, v,vids_frames_mixgauss, subacts_count, vids_frames_count, nvs, rho)
            
            subacts_order = vToPi(v[:,vid_idx])
            #z[vid_idx, :vids_frames_count[vid_idx]].sort(key = lambda x:np.where(subacts_order==x))
            z[vid_idx, :vids_frames_count[vid_idx]] = sorted(z[vid_idx, :vids_frames_count[vid_idx]], key = lambda x:np.where(subacts_order==x))
            
        #resample_rho here [for each iteration]
            
    #MAJOR #TODO: add LHS,RHS and returnables for rho
    samples[0]["z"] = z
    samples[0]["v"] = v
    samples[0]["rho"] = rho
    return samples, ntau, ntaus, nvs   
            
            
def mallows(samples,vids_frames_mixgauss, subacts_count, vids_count, vids_frames_count, tau_prior,rho_prior, iterations):
    
    tau_0 = tau_prior
    v_0 = v_prior(rho_prior, subacts_count) #used in sample_rho
    # __nu_0 = nu_prior #looks unused
     
    ntau = np.zeros([subacts_count,1], dtype = np.uint32)
    ntaus = np.zeros([1,1], dtype = np.uint32)
    #__nvs = np.zeros([subacts_count-1,1], dtype = np.uint32)
    nvs = np.sum(samples[0]["v"], axis =1 ) #TODO: verify dims
    
    samples[0]["rho"] =  rho_prior * np.ones((subacts_count-1,1)) #TODO #DOUBT: why redoing this? already initialised in initialize_subactivity_assignments.py
    
    for i in range(vids_count):
        for j in range(vids_frames_count[i]):
            act = samples[0]["z"][i][j] #TODO: verify: should be 1-indexed
            ntau[act-1] += 1
            ntaus +=1 
    
    #skipping simga and theta bcz unused
    
    #TODO finish and add params/args for sample_rho
    samples, ntau,ntaus, nvs = sample_model(samples, vids_frames_mixgauss,subacts_count, vids_count,vids_frames_count, ntau,ntaus, tau_0, nvs, iterations)
    
    #TODO evaluation functions
    
    #MAJOR #TODO include returnables from rho
    return samples
    
    
    
    
    
    
    


