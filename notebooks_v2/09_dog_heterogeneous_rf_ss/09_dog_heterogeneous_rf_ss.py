import os
import shutil

import matplotlib.pyplot as plt

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare variables.
notebook_dir = 'notebooks_v2/09_dog_heterogeneous_rf_ss'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
if os.path.exists(results_dir) and os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 6.  # centimeters
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.5
sweep_ids = [
    'nvf04nxs',  # DoG with heterogeneous RF & SS
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True,
    refresh=False)

homogeneous_indices = (runs_configs_df['place_cell_rf'] == '0.12') \
                      & (runs_configs_df['surround_scale'] == '2')
heterogeneous_indices = (runs_configs_df['place_cell_rf'] == 'Uniform( 0.06 , 0.18 )') \
                        & (runs_configs_df['surround_scale'] == 'Uniform( 1.50 , 2.50 )')
indices_to_keep = homogeneous_indices | heterogeneous_indices
runs_configs_df = runs_configs_df[indices_to_keep]

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()))

overwrite_runs_configs_df_values_with_joblib_data(
    runs_configs_df=runs_configs_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

plot_percent_runs_with_low_pos_decoding_err_pie(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir)

# Keep only networks that achieved low position decoding error.
low_pos_decoding_indices = runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm
frac_low_pos_decoding_err = low_pos_decoding_indices.mean()
print(f'Frac Low Pos Decoding Err Runs: {frac_low_pos_decoding_err}')
runs_configs_low_pos_decoding_err_df = runs_configs_df[low_pos_decoding_indices]

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

# plot_grid_score_max_vs_place_cell_rf_by_place_cell_ss(
#     runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
#     plot_dir=results_dir)

augmented_neurons_data_by_run_id_df = runs_configs_df[[
    'run_id', 'place_cell_rf', 'surround_scale']].merge(
    neurons_data_by_run_id_df,
    on='run_id',
    how='left')

plot_grid_scores_histograms_by_place_cell_rf_and_ss_homo_vs_hetero(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kdes_by_place_cell_rf_and_ss_homo_vs_hetero(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

percent_neurons_score60_above_threshold_by_run_id_df = compute_percent_neurons_score60_above_threshold_by_run_id_df(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df)

augmented_percent_neurons_score60_above_threshold_by_run_id_df = runs_configs_df[[
    'run_id', 'place_cell_rf', 'surround_scale']].merge(
    percent_neurons_score60_above_threshold_by_run_id_df,
    on='run_id',
    how='left')

plot_grid_periods_kde_by_place_cell_rf_by_place_cell_ss(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_periods_histograms_by_place_cell_rf_by_place_cell_ss(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_vs_place_cell_rf_by_place_cell_ss(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_boxen_vs_place_cell_rf_by_place_cell_ss(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

heterogeneous_indices = (augmented_neurons_data_by_run_id_df['place_cell_rf'] == 'Uniform( 0.06 , 0.18 )') \
                        & (augmented_neurons_data_by_run_id_df['surround_scale'] == 'Uniform( 1.50 , 2.50 )')

plot_rate_maps_examples_hexagons_by_score_range(
    neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df[heterogeneous_indices],
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
    plot_dir=results_dir)

plot_rate_maps_examples_squares_by_score_range(
    neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df[heterogeneous_indices],
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
    plot_dir=results_dir)

print('Finished 09_heterogeneous_receptive_field/09_heterogeneous_receptive_field.py!')
