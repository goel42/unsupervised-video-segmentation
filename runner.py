from utils.config_parse import params
from load_dataset import load_dataset
from initialize_subactivity_assignments import initialize_subactivity_assignments
from embedding.init_embedding_weights import init_embedding_weights

# LOAD DATA
activity = "coffee"
path = r"/mnt/c/Users/dcsang/Documents/code_translate/DATA/breakfast_actions/formatted_data_new/"
vid_frames_features, gt_labels, vids_frames_count, g_K = load_dataset(activity, path)
print(g_K)


#RANDOM INITIALISATION OF ACTIVITY LABELS
__rho_prior = params["gmm"]["infOut"]["rho_prior"]
samples = initialize_subactivity_assignments(vids_frames_count, g_K, __rho_prior)

#EMBEDDING PART
#dictionary of Wf and Wa
W = init_embedding_weights() #this is basically initParams

for pii in range(1, params.iterative.iterative_iters+1):
    #TODO








