import os
import shutil

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare paths.
notebook_dir = 'notebooks/14_general_dog'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')

# Remove results directory to make fresh.
# if os.path.exists(results_dir) and os.path.isdir(results_dir):
#     shutil.rmtree(results_dir)
# os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 15.
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.


runs_configs_general_dog_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=['ol70h4oy'],
    finished_only=True,
    refresh=False)

general_dog_joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_general_dog_df['run_id'].unique()),
    include_additional_data=True,  # Used to plot ratemaps
)

overwrite_runs_configs_df_values_with_joblib_data(
    runs_configs_df=runs_configs_general_dog_df,
    joblib_files_data_by_run_id_dict=general_dog_joblib_files_data_by_run_id_dict)

plot_percent_runs_with_low_pos_decoding_err_pie(
    runs_configs_df=runs_configs_general_dog_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

# Keep only networks that achieved low position decoding error.
low_pos_decoding_indices = runs_configs_general_dog_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm
frac_low_pos_decoding_err = low_pos_decoding_indices.mean()
print(f'Frac Low Pos Decoding Err Runs: {frac_low_pos_decoding_err}')
runs_configs_general_dog_df = runs_configs_general_dog_df[low_pos_decoding_indices]

general_dog_neurons_data_by_run_id_df = convert_joblib_files_data_to_neurons_data_df(
    joblib_files_data_by_run_id_dict=general_dog_joblib_files_data_by_run_id_dict)

plot_rate_maps_examples_hexagons_by_score_range(
    neurons_data_by_run_id_df=general_dog_neurons_data_by_run_id_df,
    joblib_files_data_by_run_id_dict=general_dog_joblib_files_data_by_run_id_dict,
    plot_dir=results_dir)

# Add max grid scores by run ID.
max_grid_scores_by_run_id_df = general_dog_neurons_data_by_run_id_df.groupby('run_id').agg(
    score_60_by_neuron_max=('score_60_by_neuron', 'max'),
    score_90_by_neuron_max=('score_90_by_neuron', 'max')).reset_index()

runs_configs_general_dog_df = runs_configs_general_dog_df.merge(
    max_grid_scores_by_run_id_df,
    on='run_id',
    how='left')

# Save max grid score by key parameters
runs_configs_general_dog_df[['activation', 'place_cell_alpha_e',
                             'place_cell_alpha_i', 'place_cell_rf',
                             'score_60_by_neuron_max',
                             'score_90_by_neuron_max']].to_csv(
    os.path.join(data_dir, 'runs_configs_with_max_scores.csv'),
    index=False
)

# plot_rate_maps_by_run_id(
#     neurons_data_by_run_id_df=neurons_data_by_run_id_df,
#     joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
#     plot_dir=results_dir)

plot_rate_maps_examples_squares_by_score_range(
    neurons_data_by_run_id_df=general_dog_neurons_data_by_run_id_df,
    joblib_files_data_by_run_id_dict=general_dog_joblib_files_data_by_run_id_dict,
    plot_dir=results_dir,)


def convert_place_field_values_to_human_readable_place_field_values(row: pd.Series):
    place_field_values = row['place_field_values']
    if place_field_values == 'general_difference_of_gaussians':
        replacement_place_field_values = 'General DoG'
    else:
        raise ValueError('There should be no other place field values in this dataframe!'
                         f'What is {place_field_values} doing here')
    return replacement_place_field_values


runs_configs_general_dog_df['place_field_values'] = runs_configs_general_dog_df.apply(
    convert_place_field_values_to_human_readable_place_field_values,
    axis=1)

augmented_neurons_data_by_run_id_df = runs_configs_general_dog_df[[
    'run_id', 'place_field_values']].merge(
    general_dog_neurons_data_by_run_id_df,
    on='run_id',
    how='left')

plot_grid_scores_kdes_survival_functions_by_human_readable_sweep(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir,
    figsize=(8, 6))

plot_grid_scores_histograms_by_place_field_values(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kdes_by_place_field_values(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)


print('Finished 14_general_dog/14_general_dog.py!')
