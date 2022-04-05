import os

from mec_hpc_investigations.models.analyze import compute_minima_performance_metrics_from_runs_histories, \
    download_wandb_project_runs_configs, download_wandb_project_runs_histories
from mec_hpc_investigations.models.plot import *


# Declare
notebook_dir = 'notebooks/00_broad_sweep_results'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
low_pos_decoding_err_threshold = 6.
grid_score_d60_threshold = 1.2
grid_score_d90_threshold = 1.5
# sweep_ids = [
#     '5bpvzhfh',  # Position + MSE
#     'ni9i0dfp',  # Gaussian + global norm + cross entropy + sweep hyperparameters, holding rf fixed
#     'xqyfdt1v',  # Gaussian + global norm + cross entropy + sweep rf, holding hyperparameters fixed
#     'sp2hvkth',  # DoG + global norm + cross entropy + sweep hyperparameters, holding rf fixed
#     '2cworubi',  # DoG + global norm + cross entropy + sweep rf, holding hyperparameters fixed
#     # 'y40eqafz',  # G + global norm + cross entropy + wide sweep of rf (but will rerun)
# ]

sweep_ids = [
    '2vw5jbim',  # 01: Cartesian + MSE
    'qu0mobjm',  # 03: G+Global+CE, sweeping most hyperparameters
    '8rvghgz1',  # 04: G+Global+CE, sweeping RF from 0.01m to 2.0m
    'ea06fmvq',  # 05: G+Global+CE, small RF (0.01, 0.025, 0.05), training for exceptionally long
]


os.makedirs(notebook_dir, exist_ok=True)

#
runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True)

runs_configs_df = runs_configs_df[runs_configs_df['optimizer'] != 'sgd'].copy()


def sweep_to_run_group(row: pd.Series):
    if row['Sweep'] == '2vw5jbim':
        # 01: Cartesian + MSE
        run_group = 'Cartesian\nMSE'
    elif row['Sweep'] == '':
        # 02: Polar + MSE
        run_group = 'Polar\nGeodesic'
        raise NotImplementedError
    elif row['Sweep'] == 'qu0mobjm':
        # 03: G+Global+CE, sweeping most hyperparameters
        run_group = 'Gaussian\nCE\nGlobal\nAll Hyper'
    elif row['Sweep'] == '8rvghgz1':
        # 04: G+Global+CE, sweeping RF from 0.01m to 2.0m
        run_group = 'Gaussian\nCE\nGlobal\nRF\nTrain 5x'
    elif row['Sweep'] == 'ea06fmvq':
        # 04: G+Global+CE, sweeping RF from 0.01m to 2.0m
        run_group = 'Gaussian\nCE\nGlobal\nRF\nTrain 100x'
    else:
        # run_group = f"{row['place_field_loss']}\n{row['place_field_values']}\n{row['place_field_normalization']}"
        raise ValueError
    return run_group


runs_configs_df['run_group'] = runs_configs_df.apply(
    sweep_to_run_group,
    axis=1)

runs_histories_df = download_wandb_project_runs_histories(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids)

minima_performance_metrics = compute_minima_performance_metrics_from_runs_histories(
    runs_histories_df=runs_histories_df,
)

runs_performance_df = runs_configs_df[[
    'run_id', 'run_group', 'place_field_loss', 'place_field_values', 'place_field_normalization']].merge(
        minima_performance_metrics,
        on='run_id',
        how='left')

plot_pos_decoding_err_vs_run_group(
    runs_performance_df=runs_performance_df,
    plot_dir=results_dir)

plot_percent_low_decoding_err_vs_run_group(
    runs_performance_df=runs_performance_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold=low_pos_decoding_err_threshold)

plot_pos_decoding_err_vs_max_grid_score_by_run_group(
    runs_performance_df=runs_performance_df,
    plot_dir=results_dir)

plot_max_grid_score_given_low_pos_decoding_err_vs_run_group(
    runs_performance_df=runs_performance_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold=low_pos_decoding_err_threshold)

plot_percent_have_grid_cells_given_low_pos_decoding_err_vs_run_group(
    runs_performance_df=runs_performance_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold=low_pos_decoding_err_threshold,
    grid_score_d60_threshold=grid_score_d60_threshold,
    grid_score_d90_threshold=grid_score_d90_threshold,
)

print('Finished!')
