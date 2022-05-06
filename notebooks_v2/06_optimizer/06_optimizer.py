import os

import pandas as pd

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare variables.
notebook_dir = 'notebooks_v2/06_optimizer'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold = 6.
grid_score_d60_threshold = 0.85
grid_score_d90_threshold = 1.5
sweep_ids = [
    'lgaz57h1',  # Adam and RMSProp optimizers
    'v5gndu30',  # SGD & Adagrad optimizers
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True,
    refresh=True)

# Keep only networks that achieved low position decoding error.
low_pos_decoding_indices = runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold
print(f'Frac Low Pos Decoding Err Runs: {low_pos_decoding_indices.mean()}')
runs_configs_df = runs_configs_df[low_pos_decoding_indices]

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()))

overwrite_run_config_df_values_with_joblib_data(
    runs_configs_df=runs_configs_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

plot_loss_min_vs_optimizer(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir
)

plot_pos_decoding_err_min_vs_optimizer(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir,
)

neurons_data_by_run_id_df = convert_joblib_files_data_to_neurons_data_df(
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

augmented_neurons_data_by_run_id_df = runs_configs_df[[
    'run_id', 'optimizer']].merge(
    neurons_data_by_run_id_df,
    on='run_id',
    how='left')

plot_grid_score_vs_optimizer(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir,
)

max_grid_scores_by_run_id_df = augmented_neurons_data_by_run_id_df.groupby('run_id').agg(
    score_60_by_neuron_max=('score_60_by_neuron', 'max'),
    score_90_by_neuron_max=('score_90_by_neuron', 'max'),
    optimizer=('optimizer', 'first'),
).reset_index()

plot_grid_score_max_vs_optimizer(
    max_grid_scores_by_run_id_df=max_grid_scores_by_run_id_df,
    plot_dir=results_dir,
)

# plot_max_grid_score_vs_num_grad_steps_by_optimizer(
#     runs_augmented_histories_df=runs_augmented_histories_df,
#     plot_dir=results_dir,
#     grid_score_d60_threshold=grid_score_d60_threshold,
#     grid_score_d90_threshold=grid_score_d90_threshold,
# )
#
# plot_loss_vs_num_grad_steps_by_optimizer(
#     runs_augmented_histories_df=runs_augmented_histories_df,
#     plot_dir=results_dir)
#
# plot_pos_decoding_err_vs_num_grad_steps_by_optimizer(
#     runs_augmented_histories_df=runs_augmented_histories_df,
#     plot_dir=results_dir)


print('Finished!')
