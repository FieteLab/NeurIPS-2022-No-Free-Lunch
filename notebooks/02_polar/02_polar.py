import os
import shutil

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare paths.
notebook_dir = 'notebooks/02_polar'
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
    'm10yfzgz',     # High Dim Polar
    '2gz0ks4o',     # High Dim Polar Random
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True,
    refresh=False)

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()),
    include_additional_data=True)

# overwrite_runs_configs_df_values_with_joblib_data(
#     runs_configs_df=runs_configs_df,
#     joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

plot_percent_runs_with_low_pos_decoding_err_pie(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir)

# Keep only networks that achieved low position decoding error.
low_pos_decoding_indices = runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm
print(f'Frac Low Pos Decoding Err Runs: {low_pos_decoding_indices.mean()}')
runs_configs_df = runs_configs_df[low_pos_decoding_indices]

neurons_data_by_run_id_df = convert_joblib_files_data_to_neurons_data_df(
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

max_grid_scores_by_run_id_df = neurons_data_by_run_id_df.groupby('run_id').agg(
    score_60_by_neuron_max=('score_60_by_neuron', 'max'),
    score_90_by_neuron_max=('score_90_by_neuron', 'max')).reset_index()

runs_configs_with_scores_max_df = runs_configs_df.merge(
    max_grid_scores_by_run_id_df,
    on='run_id',
    how='left')

plot_percent_runs_with_grid_cells_pie(
    runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
    plot_dir=results_dir,)

plot_percent_runs_with_grid_cells_vs_grid_score_threshold(
    runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
    plot_dir=results_dir)

plot_rate_maps_examples_hexagons_by_score_range(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
    plot_dir=results_dir)

plot_percent_runs_with_grid_cells_vs_grid_score_threshold_n_units_threshold(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kde(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kde_cdf(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kde_survival_function(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    plot_dir=results_dir)

print('Finished 02_polar/02_polar.py!')
