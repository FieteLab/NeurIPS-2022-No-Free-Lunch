import os
import shutil

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare paths.
notebook_dir = 'notebooks_v2/00_all_sweeps'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
if os.path.exists(results_dir) and os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 20.
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.5

sweep_ids = [
    '26gn9pfh',     # Cartesian + MSE
    # 'vndf9snd',     # Polar
    '7li410k6',     # G+Global+CE, sweeping non-RF hyperparameters
    # # '',  # G+Global+CE, sweeping RF from 0.01m to 2.0m
    'amk6dohd',     # DoG+Global+CE, sweeping non-RF hyperparameters Part 1
    '822u9q9v',     # DoG+Global+CE, sweeping non-RF hyperparameters Part 2
    'yzszqr74',     # DoG+Global+CE, sweeping only RF hyperparameter
    'acmd4be7',     # DoG+Global+CE, sweeping only SS hyperparameter
    'nvf04nxs',     # DoG+Global+CE, heterogeneous scales
    # 'rbrvuf2g',     # DoG+Global+CE, multiple fields Part 1
    # 'wnmp7nx0',     # DoG+Global+CE, multiple fields Part 2
    # '56legweh',     # DoG+Global+CE, multiple fields Part 3
    # 'lwalddwy',     # DoG+Global+CE, multiple fields Part 4
    'pou4d9s0',     # Softmax(Diff of Squared distances)
    '8aik21gt',     # True DoG
]

#
runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True,
    refresh=True)


# Add human-readable sweep
def convert_sweep_to_human_readable_sweep(row: pd.Series):
    sweep_id = row['Sweep']
    if sweep_id in {'26gn9pfh'}:
        human_readable_sweep = 'Cartesian\nMSE'
    elif sweep_id in {'vndf9snd'}:
        human_readable_sweep = 'Polar\nGeodesic'
    elif sweep_id in {'7li410k6'}:
        human_readable_sweep = 'G\nHyperparams'
    elif sweep_id in {'amk6dohd', '822u9q9v'}:
        human_readable_sweep = 'DoG\nHyperparams w/o RF, SS'
    elif sweep_id in {'yzszqr74'}:
        human_readable_sweep = 'DoG\nRF ' + r'$\sigma$'
    elif sweep_id in {'acmd4be7'}:
        human_readable_sweep = 'DoG\nSS ' + r'$s$'
    elif sweep_id in {'nvf04nxs'}:
        human_readable_sweep = 'DoG\nHetero RF & SS'
    elif sweep_id in {'rbrvuf2g', 'wnmp7nx0', '56legweh', 'lwalddwy'}:
        human_readable_sweep = 'DoG\nMultiple Fields'
    elif sweep_id in {'pou4d9s0'}:
        human_readable_sweep = 'Softmax Of Diff'
    elif sweep_id in {'8aik21gt'}:
        human_readable_sweep = 'True DoG'
    else:
        # run_group = f"{row['place_field_loss']}\n{row['place_field_values']}\n{row['place_field_normalization']}"
        raise ValueError
    return human_readable_sweep


runs_configs_df['human_readable_sweep'] = runs_configs_df.apply(
    convert_sweep_to_human_readable_sweep,
    axis=1)

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()))

# overwrite_runs_configs_df_values_with_joblib_data(
#     runs_configs_df=runs_configs_df,
#     joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

# plot_percent_runs_with_low_pos_decoding_err_pie(
#     runs_configs_df=runs_configs_df,
#     plot_dir=results_dir,
#     low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

# plot_pos_decoding_err_vs_human_readable_sweep(
#     runs_configs_df=runs_configs_df,
#     plot_dir=results_dir)
#
# plot_percent_low_decoding_err_vs_human_readable_sweep(
#     runs_configs_df=runs_configs_df,
#     plot_dir=results_dir,
#     low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

# Keep only networks that achieved low position decoding error.
low_pos_decoding_indices = runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm
frac_low_position_decoding_err = low_pos_decoding_indices.mean()
print(f'Frac Low Pos Decoding Err Runs: {frac_low_position_decoding_err}')
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

# plot_pos_decoding_err_vs_max_grid_score_kde(
#     runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
#     plot_dir=results_dir)

# plot_percent_runs_with_grid_cells_pie(
#     runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
#     plot_dir=results_dir,)


# plot_pos_decoding_err_vs_max_grid_score_by_human_readable_sweep(
#     runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
#     plot_dir=results_dir)

plot_max_grid_score_given_low_pos_decoding_err_vs_human_readable_sweep(
    runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

# plot_max_grid_score_vs_activation(
#     runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
#     plot_dir=results_dir)

plot_percent_have_grid_cells_given_low_pos_decoding_err_vs_human_readable_sweep(
    runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

augmented_neurons_data_by_run_id_df = runs_configs_df[[
    'run_id', 'human_readable_sweep',]].merge(
    neurons_data_by_run_id_df,
    on='run_id',
    how='left')

plot_grid_scores_kdes_by_human_readable_sweep(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_histograms_by_human_readable_sweep(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

# plot_grid_scores_histograms_by_place_field_values(
#     augmented_neurons_data_by_run_id_df=
# )
#
#
# plot_grid_scores_kdes_by_place_field_values(
#     augmented_neurons_data_by_run_id_df=
# )

print('Finished!')
