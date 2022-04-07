import os

from mec_hpc_investigations.models.analyze import compute_minima_performance_metrics_from_runs_histories, \
    download_wandb_project_runs_configs, download_wandb_project_runs_histories
from mec_hpc_investigations.models.plot import *


# Declare
notebook_dir = 'notebooks/04_dog_relu_produces_squares'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold = 6.
grid_score_d60_threshold = 1.2
grid_score_d90_threshold = 1.5
sweep_ids = [
    '05ljtf0t',  # 05: DoG+Global+CE, sweeping others
]


runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True)


def sweep_to_run_group(row: pd.Series):
    if row['Sweep'] == '05ljtf0t':
        run_group = 'CE\nDoG\nGlobal\nOthers\nN=25'
    else:
        run_group = f"{row['place_field_loss']}\n{row['place_field_values']}\n{row['place_field_normalization']}"
    return run_group


runs_configs_df['run_group'] = runs_configs_df.apply(
    sweep_to_run_group,
    axis=1)

runs_configs_df = runs_configs_df[runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold]

runs_histories_df = download_wandb_project_runs_histories(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids)

plot_loss_over_min_loss_vs_epoch_by_run_id(
    runs_histories_df=runs_histories_df,
    plot_dir=results_dir,
)

plot_pos_decoding_err_over_min_pos_decoding_err_vs_epoch_by_run_id(
    runs_histories_df=runs_histories_df,
    plot_dir=results_dir,
)

minima_performance_metrics = compute_minima_performance_metrics_from_runs_histories(
    runs_histories_df=runs_histories_df,
)

runs_performance_df = runs_configs_df[[
    'run_id', 'run_group', 'place_field_loss', 'place_field_values',
    'place_field_normalization', 'place_cell_rf', 'activation',
    'rnn_type']].merge(
        minima_performance_metrics,
        on='run_id',
        how='left')

plot_max_grid_score_vs_place_cell_rf_by_activation(
    runs_performance_df=runs_performance_df,
    plot_dir=results_dir
)

plot_max_grid_score_90_vs_max_grid_score_60_by_activation(
    runs_performance_df=runs_performance_df,
    plot_dir=results_dir,
    grid_score_d60_threshold=grid_score_d60_threshold,
    grid_score_d90_threshold=grid_score_d90_threshold,
)

plot_participation_ratio_vs_architecture_and_activation(
    runs_performance_df=runs_performance_df,
    plot_dir=results_dir,
)

print('Finished!')
