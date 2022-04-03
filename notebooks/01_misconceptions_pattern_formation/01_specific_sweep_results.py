import os

from mec_hpc_investigations.models.analyze import compute_minima_performance_metrics_from_runs_histories, \
    download_wandb_project_runs_configs, download_wandb_project_runs_histories
from mec_hpc_investigations.models.plot import *


# Declare
plot_dir = 'notebooks/01_misconceptions_pattern_formation/results/'
low_pos_decoding_err_threshold = 6.
grid_score_d60_threshold = 1.2
grid_score_d90_threshold = 1.5
sweep_ids = [
    '8rvghgz1',  # Gaussians + Global + CE, sweeping RF from 0.01 to 2.00
]


os.makedirs(plot_dir, exist_ok=True)

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    sweep_ids=sweep_ids,
    finished_only=True)


def sweep_to_run_group(row: pd.Series):
    if row['Sweep'] == 'y40eqafz':
        run_group = 'CE\nGaussian\nGlobal\nRF=Var'
    else:
        run_group = f"{row['place_field_loss']}\n{row['place_field_values']}\n{row['place_field_normalization']}"
    return run_group


runs_configs_df['run_group'] = runs_configs_df.apply(
    sweep_to_run_group,
    axis=1)

runs_histories_df = download_wandb_project_runs_histories(
    wandb_project_path='mec-hpc-investigations',
    sweep_ids=sweep_ids)

plot_loss_over_min_loss_vs_epoch_by_run_id(
    runs_histories_df=runs_histories_df,
    plot_dir=plot_dir,
)

plot_pos_decoding_err_over_min_pos_decoding_err_vs_epoch_by_run_id(
    runs_histories_df=runs_histories_df,
    plot_dir=plot_dir,
)

minima_performance_metrics = compute_minima_performance_metrics_from_runs_histories(
    runs_histories_df=runs_histories_df,
)

runs_performance_df = runs_configs_df[[
    'run_id', 'run_group', 'place_field_loss', 'place_field_values',
    'place_field_normalization', 'place_cell_rf', 'activation']].merge(
        minima_performance_metrics,
        on='run_id',
        how='left')

plot_max_grid_score_vs_place_cell_rf_by_activation(
    runs_performance_df=runs_performance_df,
    plot_dir=plot_dir
)

print('Finished!')
