import os

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare paths.
notebook_dir = 'notebooks_v2/00_all_sweeps'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 6.
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.5

sweep_ids = [
    '26gn9pfh',  # Cartesian + MSE
    # 'rutsx042',  # G+Global+CE, sweeping non-RF hyperparameters
    # # '',  # G+Global+CE, sweeping RF from 0.01m to 2.0m
    # 'amk6dohd',  # DoG+Global+CE, sweeping non-RF hyperparameters
    # 'yzszqr74',  # DoG+Global+CE, sweeping only RF hyperparameter
    # '2yfpvx86',  # DoG+Global+CE, heterogeneous RF & SS
    # 'rbrvuf2g',  # DoG+Global+CE, multiple fields 1
    # 'wnmp7nx0',  # DoG+Global+CE, multiple fields 2
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
    if row['Sweep'] == '26gn9pfh':
        # 01: Cartesian + MSE
        human_readable_sweep = 'Cartesian\nMSE\nN=144'
    elif row['Sweep'] == '':
        # 02: Polar + MSE
        human_readable_sweep = 'Polar\nGeodesic'
        raise NotImplementedError
    elif row['Sweep'] == 'rutsx042':
        human_readable_sweep = 'Gaussian\nHyperparams w/o RF, SS)\nN='
    elif row['Sweep'] == 'amk6dohd':
        human_readable_sweep = 'DoG\nHyperparams w/o RF, SS\nN='
    elif row['Sweep'] == 'yzszqr74':
        human_readable_sweep = 'DoG\nRF\nN='
    elif row['Sweep'] == '2yfpvx86':
        human_readable_sweep = 'DoG\nHetero RF & SS\nN='
    elif row['Sweep'] == 'rbrvuf2g' or row['Sweep'] == 'wnmp7nx0':
        human_readable_sweep = 'DoG\nMultiple Fields\nN='
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

plot_percent_runs_with_low_pos_decoding_err_pie(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

plot_pos_decoding_err_vs_human_readable_sweep(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir)

plot_percent_low_decoding_err_vs_human_readable_sweep(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

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

# plot_pos_decoding_err_vs_max_grid_score_kde(
#     runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
#     plot_dir=results_dir)

plot_percent_runs_with_grid_cells_pie(
    runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
    plot_dir=results_dir,)


plot_pos_decoding_err_vs_max_grid_score_by_human_readable_sweep(
    runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
    plot_dir=results_dir)

plot_max_grid_score_given_low_pos_decoding_err_vs_human_readable_sweep(
    runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

plot_max_grid_score_vs_activation(
    runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
    plot_dir=results_dir)

plot_percent_have_grid_cells_given_low_pos_decoding_err_vs_human_readable_sweep(
    runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm,
    grid_score_d60_threshold=grid_score_d60_threshold,
    grid_score_d90_threshold=grid_score_d90_threshold)

print('Finished!')
