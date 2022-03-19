import os
import socket

ROOT_DIR = '/om2/user/rylansch/MEC-HPC-Models-Investigation'


# data dirs for mec
BASE_DIR = os.path.join(ROOT_DIR, "mec_data/")
BASE_DIR_PACKAGED = os.path.join(BASE_DIR, "packaged_data/")
CAITLIN_BASE_DIR = os.path.join(BASE_DIR, "caitlin/")
REWARD_BASE_DIR = os.path.join(BASE_DIR, "reward_data/")
CAITLIN2D_WITH_INERTIAL = os.path.join(CAITLIN_BASE_DIR, "Freely_moving_data_with_inertial_sensor.mat")
CAITLIN2D_WITHOUT_INERTIAL = os.path.join(CAITLIN_BASE_DIR, "Freely_moving_data_without_inertial_sensor.mat")
CAITLIN1D_VR = os.path.join(CAITLIN_BASE_DIR, "vr1d_files/")
CAITLIN1D_VR_PACKAGED = os.path.join(BASE_DIR_PACKAGED, "caitlin1dvr.pkl")
CAITLIN1D_VR_TRIALBATCH_DIR = os.path.join(BASE_DIR_PACKAGED, "caitlin_1d_vr_trial_batches/")
CAITLIN_BSCORES = os.path.join(CAITLIN_BASE_DIR, "border_scores/")
CAITLIN_HDSCORES = os.path.join(CAITLIN_BASE_DIR, "head_direction_scores/")

# saved model dir
BASE_DIR_MODELS = os.path.join(ROOT_DIR, "mec_models/")
BANINO_REP_DIR = os.path.join(BASE_DIR_MODELS, "banino_rep_dir/")

# saved neural fit results dir
BASE_DIR_RESULTS = os.path.join(ROOT_DIR, "mec_results/")
CAITLIN1D_VR_INTERANIMAL_CON_RESULTS = os.path.join(BASE_DIR_RESULTS, "caitlin_1d_vr_interanimal_consistencies/")
CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET = os.path.join(BASE_DIR_RESULTS, "caitlin_1d_vr_interanimal_consistencies_elasticnet/")
CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP = os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET, "map_kwargs/")
CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP_MODEL_RESULTS = os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET, "model_results/")
CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_MAP_INTERANIMAL_RESULTS = os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET, "interanimal_results/")
CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET_AGG_CONS = os.path.join(CAITLIN1D_VR_INTERANIMAL_CON_RESULTS_ELASTICNET, "agg_cons/")
CAITLIN2D_INTERANIMAL_TRAINFRAC_RESULTS = os.path.join(BASE_DIR_RESULTS, "caitlin_2d_interanimal_trainfrac/")
CAITLIN2D_INTERANIMAL_SAMPLE_RESULTS = os.path.join(BASE_DIR_RESULTS, "caitlin_2d_interanimal_sample/")
CAITLIN2D_INTERANIMAL_SAMPLE_AGG_RESULTS = os.path.join(CAITLIN2D_INTERANIMAL_SAMPLE_RESULTS, "agg_results/")
CAITLIN2D_INTERANIMAL_CC_MAP = os.path.join(BASE_DIR_RESULTS, "caitlin_2d_interanimal_cc_maps/")
CAITLIN2D_INTERANIMAL_CC_MAP_MODEL_RESULTS = os.path.join(CAITLIN2D_INTERANIMAL_CC_MAP, "model_results/")
CAITLIN2D_MODEL_BORDERGRID_RESULTS = os.path.join(CAITLIN2D_INTERANIMAL_CC_MAP, "model_bordergridscores/")
MODEL_CV_RESULTS_CAITLIN2D = os.path.join(BASE_DIR_RESULTS, "model_cv_results_caitlin_2d/")
OFREWARD_COMBINED_INTERANIMAL_SAMPLE_RESULTS = os.path.join(BASE_DIR_RESULTS, "ofreward_combined_interanimal_sample/")
OFREWARD_COMBINED_INTERANIMAL_CC_MAP = os.path.join(BASE_DIR_RESULTS, "ofreward_combined_interanimal_cc_maps/")
OFREWARD_COMBINED_INTERANIMAL_CC_MAP_MODEL_RESULTS = os.path.join(OFREWARD_COMBINED_INTERANIMAL_CC_MAP, "model_results/")
CAITLINHPC_INTERANIMAL_SAMPLE_RESULTS = os.path.join(BASE_DIR_RESULTS, "caitlinhpc_interanimal_sample/")

# data dirs for hpc
BASE_DIR_HPC = os.path.join(ROOT_DIR, "hpc_data/")
CAITLIN_BASE_DIR_HPC = os.path.join(BASE_DIR_HPC, "caitlin/")
CAITLIN2D_PLACE_CELL = os.path.join(CAITLIN_BASE_DIR_HPC, "place_cells/")
CAITLIN2D_HPC = os.path.join(CAITLIN_BASE_DIR_HPC, "hpc_all/")
BASE_DIR_PLACE_CELL_RESULTS = os.path.join(ROOT_DIR, "place_cell_results/")
BASE_DIR_HPC_RESULTS = os.path.join(ROOT_DIR, "hpc_results/")
HPC_INTERANIMAL_CC_MAP = os.path.join(BASE_DIR_HPC_RESULTS, "interanimal_cc_maps/")
HPC_INTERANIMAL_CC_MAP_MODEL_RESULTS = os.path.join(HPC_INTERANIMAL_CC_MAP, "model_results/")