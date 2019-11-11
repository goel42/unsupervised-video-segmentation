import numpy as np
#DOUBT: tried my best to test it, would suggest you test it too. 
# DOUBT: for a given inversion count, will the output order be unique?

#this function takes a horizontal list as input instead of vertical list as in the original code 

def vToPi(v):
    
    #to account for 0-based indexing in python, just subracted wherever container is accessed
    sub=1
    
    pi_len = len(v)+1
    pi = np.zeros(pi_len)
    pi[0] = pi_len
    
    for i in range(pi_len-2,0-sub,-1):
        for j in range(pi_len-i-1,v[i+1-sub]+1-sub, -1):
            pi[j+1 -sub] = pi[j-sub]
        pi[v[i+1-sub] +1 -sub] = i+1
    return pi


    