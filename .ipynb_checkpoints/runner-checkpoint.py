from utils.config_parse import params
from load_dataset import load_dataset
from initialize_subactivity_assignments import initialize_subactivity_assignments
from embedding.init_embedding_weights import init_embedding_weights
from embedding.optimise_embedding_weights import optimise_embedding_weights

#subacts_count == g_K
#vids_frames_gt_label == gt_labels
#latent space transformation Weight: W {Wf, Wa}

# LOAD DATA
activity = "coffee"
path = r"/mnt/c/Users/dcsang/Documents/code_translate/DATA/breakfast_actions/formatted_data_new/"
vids_frames_features, vids_frames_gtlabel, vids_frames_count, subacts_count = load_dataset(activity, path)

#RANDOM INITIALISATION OF ACTIVITY LABELS
__rho_prior = params["gmm"]["infOut"]["rho_prior"]
samples = initialize_subactivity_assignments(vids_frames_count, subacts_count, __rho_prior)

###EMBEDDING PART
path2embeddingdata_folder = "/mnt/c/Users/dcsang/Documents/code_translate/BREAKFAST_EXP/breakfast_iter/intermediate/INTERMEDIATE_DATA/FINAL_breakfast_iter/coffee/experiment_k_6/v_0_s_50_e_1_nGauss_3word2vec/rand_iter_1/iterative_1"
W = init_embedding_weights() #this is  initParams of original code
W = optimise_embedding_weights(path2embeddingdata_folder, W)
#TODO: samples has 0-padding for vids with frames less than max_frames_count,
#and iteration_seglabels is basically creating a DS with variable row sizes and removing this padding
#i am going ahead with samples, will change later if necessary

###GAUSSIAN MIXTURE PART


###INFERENCE AND MALLOWS MODEL












