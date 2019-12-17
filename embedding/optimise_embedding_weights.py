from embedding.defrag_run import defrag_run
def optimise_embedding_weights(path2embeddingdata, W):
    
    
    W = defrag_run(path2embeddingdata, W)
    return W