#vids_frames_label could either be ground truth labels or the labels estimated by GMM or random

#TODO write code for saving train test val splits to file
def prep_data(path, vids_frames_features, vids_frames_count, vids_frames_label, subacts_count):
    pval = #TODO
    
    #TODO do we need number_canonical and number_shuffle?
    #TODO
    labels_features= get_word2vec()
    
    frames_count = sum(vids_frames_count) #TODO verify output
    frames_all = []
    labels_all = []
    
    for frame_features, label in zip(vids_frames_features, vids_frames_label):
        
        
        
    
        
    
    
