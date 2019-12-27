# -*- coding: utf-8 -*-
import random

def sample_normalised(logprobs, count):
    
    __rand = #random.uniform(0,1) #TODO in C code, all random nums are first collected together in g_unifrnd array and then passed around one by one. if i simply gnerate each random num separately like this. isnt it the same thing?
    
    __max_logprob = np.max(logprobs)
    
    for i in range(count):
        logprobs[i] = math.exp(logprobs[i] - __max_logprob)
    
    __sum = np.sum(logprobs)
    
    __scaled_random = __rand * __sum
    
    __sum = 0.0
    for i in range(count):
        __sum += logprobs[i]
        if (__scaled_random < __sum):
            return i
   
    return math.round(math.floor(__rand * count)) #TODO: why round over floor? output of floor would alr be int

    
    