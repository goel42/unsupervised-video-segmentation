# -*- coding: utf-8 -*-
import numpy as np 

#TODO: wordmap of original code is not used. skipping it here
def load_word_vectors():
    #word_dim = 200
    path = r"C:\Users\dcsang\Documents\code_translate\LIBS\word2vec\wordvecs_200d.mat"
    vecs = hdf5storage.loadmat(path)
 
    #TODO: following is false in original code so skipping it for now. 
    #this is for normalising the word2vecs
    #if params["l2norm"]:
        #TODO
    
    return vecs["oWe"], vecs["vocab"]
    
    
def load_word2vecs_from_txt():
    embeddings_dict = {}
    path = r"C:\Users\dcsang\Documents\code_translate\LIBS\word2vec\vectors_200d.txt"
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


#TODO: skipped stoi part, is it used later?
def get_label_word2vec(path, subacts_count):
#Given the labels returns the word2vec features of the labels
#path2embeddingdata: the directory formatted data will be saved in
#n_subactivities : total number of sub-activities for the current activity.
    
    #TODO: take from params, skipping for now
    #label_decimal = #TODO
    label_decimal =1
    
    word2vecs = load_word2vecs_from_txt()
    vocab = list(word2vecs.keys())
    labels_str = [str(i*label_decimal) for i in range(1, subacts_count+1)]
    labels_word2vecs = {}
    
    for label_str in labels_str:
        vec_idx = vocab.index(label_str)
        vec = word2vecs[label_str]
        label_info ={ "str": label_str,
                     "idx": vec_idx,
                     "vec": vec}
        labels_word2vecs[int(label_str)] = label_info
    
    #TODO save these label word2vecs in path
    return labels_word2vecs