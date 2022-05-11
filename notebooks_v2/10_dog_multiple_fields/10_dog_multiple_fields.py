import os

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare variables.
notebook_dir = 'notebooks_v2/10_dog_multiple_fields'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold = 6.
grid_score_d60_threshold = 0.85
grid_score_d90_threshold = 1.5
sweep_ids = [

]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True,
    refresh=True)

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()))

overwrite_runs_configs_df_values_with_joblib_data(
    runs_configs_df=runs_configs_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

# Keep only networks that achieved low position decoding error.
low_pos_decoding_indices = runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold
print(f'Frac Low Pos Decoding Err Runs: {np.round(low_pos_decoding_indices.mean(), 3)}')
runs_configs_df = runs_configs_df[low_pos_decoding_indices]

neurons_data_by_run_id_df = convert_joblib_files_data_to_neurons_data_df(
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

augmented_neurons_data_by_run_id_df = runs_configs_df[[
    'run_id', 'n_place_fields_per_cell']].merge(
    neurons_data_by_run_id_df,
    on='run_id',
    how='left')

# augmented_neurons_data_by_run_id_df.groupby('run_id')['period_per_cell']

plot_grid_periods_histograms_by_place_cell_rf(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_periods_kde_by_place_cell_rf(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_num_grid_cells_by_place_cell_rf(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

max_grid_scores_by_run_id_df = augmented_neurons_data_by_run_id_df.groupby('run_id').agg(
    score_60_by_neuron_max=('score_60_by_neuron', 'max'),
    score_90_by_neuron_max=('score_90_by_neuron', 'max'),
    place_cell_rf=('place_cell_rf', 'first')).reset_index()

plot_grid_score_max_as_dots_vs_place_cell_rf(
    max_grid_scores_by_run_id_df=max_grid_scores_by_run_id_df,
    plot_dir=results_dir,)

print('Finished 08_dog_receptive_field/08_dog_receptive_field.py!')
