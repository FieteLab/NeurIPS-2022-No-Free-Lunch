import os
import shutil

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare paths.
notebook_dir = 'notebooks_v2/04_DoG_vs_DoS'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')

# Remove results directory to make fresh.
if os.path.exists(results_dir) and os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 15.
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.


runs_configs_true_dog_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=['nisioabg'],  # True DoG
    finished_only=True,
    refresh=True)

true_dog_joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_true_dog_df['run_id'].unique()),
    include_additional_data=True)

overwrite_runs_configs_df_values_with_joblib_data(
    runs_configs_df=runs_configs_true_dog_df,
    joblib_files_data_by_run_id_dict=true_dog_joblib_files_data_by_run_id_dict)

# plot_percent_runs_with_low_pos_decoding_err_pie(
#     runs_configs_df=runs_configs_true_dog_df,
#     plot_dir=results_dir,
#     low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

# Keep only networks that achieved low position decoding error.
low_pos_decoding_indices = runs_configs_true_dog_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm
frac_low_pos_decoding_err = low_pos_decoding_indices.mean()
print(f'Frac Low Pos Decoding Err Runs: {frac_low_pos_decoding_err}')
runs_configs_true_dog_df = runs_configs_true_dog_df[low_pos_decoding_indices]

true_dog_neurons_data_by_run_id_df = convert_joblib_files_data_to_neurons_data_df(
    joblib_files_data_by_run_id_dict=true_dog_joblib_files_data_by_run_id_dict)

# plot_rate_maps_examples_hexagons_by_score_range(
#     neurons_data_by_run_id_df=true_dog_neurons_data_by_run_id_df,
#     joblib_files_data_by_run_id_dict=true_dog_joblib_files_data_by_run_id_dict,
#     plot_dir=results_dir)

# plot_rate_maps_by_run_id(
#     neurons_data_by_run_id_df=neurons_data_by_run_id_df,
#     joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
#     plot_dir=results_dir)

# plot_rate_maps_examples_squares_by_score_range(
#     neurons_data_by_run_id_df=true_dog_neurons_data_by_run_id_df,
#     joblib_files_data_by_run_id_dict=true_dog_joblib_files_data_by_run_id_dict,
#     plot_dir=results_dir)

# Load DoS
runs_configs_dog_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=['vxbwdefk'],
    finished_only=True,
    refresh=True)

# Keep only the ideal DoS runs.
runs_configs_dog_df = runs_configs_dog_df[
    (runs_configs_dog_df['place_cell_rf'] == 0.12) &
    (runs_configs_dog_df['surround_scale'] == 2.0) &
    (runs_configs_dog_df['optimizer'] == 'adam')]

# Merge the two dataframes.
runs_configs_df = pd.concat([runs_configs_true_dog_df, runs_configs_dog_df ]).reset_index()

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()))

neurons_data_by_run_id_df = convert_joblib_files_data_to_neurons_data_df(
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)


def convert_place_field_values_to_human_readable_place_field_values(row: pd.Series):
    place_field_values = row['place_field_values']
    if place_field_values == 'difference_of_gaussians':
        replacement_place_field_values = '"DoG" (DoS)'
    elif place_field_values == 'true_difference_of_gaussians':
        replacement_place_field_values = 'True DoG'
    else:
        # run_group = f"{row['place_field_loss']}\n{row['place_field_values']}\n{row['place_field_normalization']}"
        raise ValueError
    return replacement_place_field_values


runs_configs_df['place_field_values'] = runs_configs_df.apply(
    convert_place_field_values_to_human_readable_place_field_values,
    axis=1)

augmented_neurons_data_by_run_id_df = runs_configs_df[[
    'run_id', 'place_field_values']].merge(
    neurons_data_by_run_id_df,
    on='run_id',
    how='left')

plot_grid_scores_histograms_by_place_field_values(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kdes_by_place_field_values(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

# max_grid_scores_by_run_id_df = augmented_neurons_data_by_run_id_df.groupby('run_id').agg(
#     score_60_by_neuron_max=('score_60_by_neuron', 'max'),
#     score_90_by_neuron_max=('score_90_by_neuron', 'max'),
#     place_cell_rf=('place_field_values', 'first')).reset_index()
#
# runs_configs_with_scores_max_df = runs_configs_true_dog_df.merge(
#     max_grid_scores_by_run_id_df,
#     on='run_id',
#     how='left')
#
# plot_percent_runs_with_grid_cells_pie(
#     runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
#     plot_dir=results_dir, )

print('Finished 04_DoG_vs_DoS/04_DoG_vs_DoS.py!')
