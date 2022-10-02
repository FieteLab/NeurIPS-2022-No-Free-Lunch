import os
import shutil

import matplotlib.pyplot as plt

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare variables.
notebook_dir = 'notebooks/13_dos_multiple_fields_and_scales'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
if os.path.exists(results_dir) and os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 10.  # centimeters
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.5
sweep_ids = [
    'bav6z2py',     # DoS Ideal
    'lk012xp8',     # DoS Multi-field, Multi-Scale Part 1
    '2lj5ngjz',     # DoS Multi-field, Multi-Scale Part 2
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True,
    refresh=False)


single_scale_single_field_indices = (runs_configs_df['n_place_fields_per_cell'] == '1') & \
                                    (runs_configs_df['place_cell_rf'] == '0.12') \
                                    & (runs_configs_df['surround_scale'] == '2')
multi_scale_multi_field_indices = (runs_configs_df['n_place_fields_per_cell'] == 'Poisson( 3.0 )') & \
                                  (runs_configs_df['place_cell_rf'] == 'Uniform( 0.06 , 1.0 )') \
                                  & (runs_configs_df['surround_scale'] == 'Uniform( 1.25 , 4.50 )')

print(f"Num single-scale & single-field runs: {sum(single_scale_single_field_indices)}")
print(f"Num multi-scale & multi-field runs: {sum(multi_scale_multi_field_indices)}")

indices_to_keep = single_scale_single_field_indices | multi_scale_multi_field_indices
runs_configs_df = runs_configs_df[indices_to_keep]


# Add human-readable sweep
def convert_sweeps_to_human_readable_sweep(row: pd.Series):
    sweep_id = row['Sweep']
    if sweep_id in {'bav6z2py'}:
        human_readable_sweep = 'DoS\nSingle Field\nSingle Scales'
    elif sweep_id in {'lk012xp8', '2lj5ngjz'}:
        human_readable_sweep = 'DoS\nMultiple Fields\nMultiple Scales'
    else:
        # run_group = f"{row['place_field_loss']}\n{row['place_field_values']}\n{row['place_field_normalization']}"
        raise ValueError
    return human_readable_sweep


runs_configs_df['human_readable_sweep'] = runs_configs_df.apply(
    convert_sweeps_to_human_readable_sweep,
    axis=1)

# Append the number of runs per human-readable sweep to the human-readable sweep.
num_runs_per_human_readable_sweep = runs_configs_df.groupby('human_readable_sweep').size().to_dict()
print(f"Num Runs per Human Readable Sweep: {num_runs_per_human_readable_sweep}")
runs_configs_df['human_readable_sweep'] = runs_configs_df.apply(
    lambda row: row['human_readable_sweep'] + "\nN = " + str(num_runs_per_human_readable_sweep[row['human_readable_sweep']]),
    axis=1)

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()),
    include_additional_data=True)

overwrite_runs_configs_df_values_with_joblib_data(
    runs_configs_df=runs_configs_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

plot_percent_runs_with_low_pos_decoding_err_pie(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir)

plot_percent_low_decoding_err_vs_human_readable_sweep(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

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

augmented_neurons_data_by_run_id_df = runs_configs_df[[
    'run_id', 'place_cell_rf', 'surround_scale', 'human_readable_sweep']].merge(
    neurons_data_by_run_id_df,
    on='run_id',
    how='left')

plot_grid_scores_histograms_by_human_readable_sweep(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kdes_by_human_readable_sweep(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kdes_cdfs_by_human_readable_sweep(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kdes_survival_functions_by_human_readable_sweep(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir,
    figsize=(8, 6))

percent_neurons_score60_above_threshold_by_run_id_df = compute_percent_neurons_score60_above_threshold_by_run_id_df(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df)

augmented_percent_neurons_score60_above_threshold_by_run_id_df = runs_configs_df[[
    'human_readable_sweep', 'run_id', 'n_place_fields_per_cell', 'place_cell_rf',
    'surround_scale']].merge(
    percent_neurons_score60_above_threshold_by_run_id_df,
    on='run_id',
    how='left')

plot_rate_maps_examples_hexagons_by_score_range(
    neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df[
        augmented_neurons_data_by_run_id_df['human_readable_sweep'] == "DoS\nMultiple Fields\nMultiple Scales\nN = 6"
    ],
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
    plot_dir=results_dir)

plot_rate_maps_examples_hexagons_by_score_sorted(
    neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df[
        augmented_neurons_data_by_run_id_df['human_readable_sweep'] == "DoS\nMultiple Fields\nMultiple Scales\nN = 6"
        ],
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
    plot_dir=results_dir
)

print('Finished 13_dos_multiple_fields_and_scales/13_dos_multiple_fields_and_scales.py!')
