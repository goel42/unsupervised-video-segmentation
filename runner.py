from utils.config_parse import params
from load_dataset import load_dataset
from initialize_subactivity_assignments import initialize_subactivity_assignments
from embedding.init_embedding_weights import init_embedding_weights
from embedding.optimise_embedding_weights import optimise_embedding_weights
from embedding.feature_extraction import extract_features
from gauss_mix.compute_gauss_mix import compute_gauss_mix
from utils.load_word_vectors import get_label_word2vec

#from embedding.optimise_embedding_weights import optimise_embedding_weights

#subacts_count == g_K
#vids_frames_gt_label == gt_labels
#latent space transformation Weight: W {Wf, Wa}

# LOAD DATA
activity = "coffee"
path = r"C:/Users/dcsang/Documents/code_translate/DATA/breakfast_actions/formatted_data_new/"
vids_frames_features, vids_frames_gtlabel, vids_frames_count, subacts_count = load_dataset(activity, path)

#RANDOM INITIALISATION OF ACTIVITY LABELS
__rho_prior = params["gmm"]["infOut"]["rho_prior"]
samples = initialize_subactivity_assignments(vids_frames_count, subacts_count, __rho_prior)

###GET VECS/FEATURES FOR LABELS (aka TextFeat in original code)
#TODO do we need number_canonical and number_shuffle? 
#made changes in this structure. in orig code this code is present within fs_prep_embedding_data
labels_features= get_label_word2vec(path, subacts_count)

###EMBEDDING PART
path2embeddingdata_folder = r"C:/Users/dcsang/Documents/code_translate/BREAKFAST_EXP/breakfast_iter/intermediate/INTERMEDIATE_DATA/FINAL_breakfast_iter/coffee/experiment_k_6/v_0_s_50_e_1_nGauss_3word2vec/rand_iter_1/iterative_1"
W = init_embedding_weights() #this is  initParams of original code
W = optimise_embedding_weights(path2embeddingdata_folder, vids_frames_features, 
                               vids_frames_count, subacts_count, samples[0]["z"],W, labels_features)
#TODO: samples has 0-padding for vids with frames less than max_frames_count,
#and iteration_seglabels is basically creating a DS with variable row sizes and removing this padding
#i am going ahead with samples, will change later if necessary

###FEATURE EXTRACTION
X_probs = extract_features(labels_features, vids_frames_features, W)

print("#########################################bro")
###GAUSSIAN MIXTURE PART
X_probs_gauss_mix = compute_gauss_mix(samples[0]["z"], vids_frames_count, X_probs, subacts_count)

###INFERENCE AND MALLOWS MODEL
#sample_model()













