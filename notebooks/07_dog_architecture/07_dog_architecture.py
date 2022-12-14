import os
import shutil

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare variables.
notebook_dir = 'notebooks/07_dog_architecture'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
if os.path.exists(results_dir) and os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 10.
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.5
sweep_ids = [
    'amk6dohd',  # DoG+Global+CE, RNN, LSTM, GRU, UGRNN
    '822u9q9v',
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True,
    refresh=True)

# Keep only ReLU & Tanh nonlinearities.
runs_configs_df = runs_configs_df[runs_configs_df['activation'] == 'relu']

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()))

overwrite_runs_configs_df_values_with_joblib_data(
    runs_configs_df=runs_configs_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

# Keep only networks that achieved low position decoding error.
low_pos_decoding_indices = runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm
print(f'Frac Low Pos Decoding Err Runs: {low_pos_decoding_indices.mean()}')
runs_configs_df = runs_configs_df[low_pos_decoding_indices]

plot_loss_min_vs_architecture(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir)

plot_pos_decoding_err_min_vs_architecture(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir)

neurons_data_by_run_id_df = convert_joblib_files_data_to_neurons_data_df(
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

augmented_neurons_data_by_run_id_df = runs_configs_df[[
    'run_id', 'rnn_type']].merge(
    neurons_data_by_run_id_df,
    on='run_id',
    how='left')

plot_grid_scores_vs_architecture(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir,
)

max_grid_scores_by_run_id_df = augmented_neurons_data_by_run_id_df.groupby('run_id').agg(
    score_60_by_neuron_max=('score_60_by_neuron', 'max'),
    score_90_by_neuron_max=('score_90_by_neuron', 'max'),
    rnn_type=('rnn_type', 'first')).reset_index()

plot_grid_score_max_vs_architecture(
    max_grid_scores_by_run_id_df=max_grid_scores_by_run_id_df,
    plot_dir=results_dir)

plot_grid_score_max_90_vs_grid_score_max_60_by_architecture(
    max_grid_scores_by_run_id_df=max_grid_scores_by_run_id_df,
    plot_dir=results_dir)

print('Finished 07_architecture/07_architecture.py!')
