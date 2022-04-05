import os

from mec_hpc_investigations.models.analyze import compute_minima_performance_metrics_from_runs_histories, \
    download_wandb_project_runs_configs, download_wandb_project_runs_histories
from mec_hpc_investigations.models.plot import *


# Declare
notebook_dir = 'notebooks/02_exceptionally_long_gaussian_training'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
low_pos_decoding_err_threshold = 6.
grid_score_d60_threshold = 1.2
grid_score_d90_threshold = 1.5

sweep_ids = [
    'ea06fmvq',  # 05: G+Global+CE, small RF (0.01, 0.025, 0.05), training for exceptionally long
]


os.makedirs(notebook_dir, exist_ok=True)

#
runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=False)


def sweep_to_run_group(row: pd.Series):
    if row['Sweep'] == 'ea06fmvq':
        # 05: G+Global+CE, sweeping RF from 0.01m to 0.05m, training 100x
        run_group = 'Gaussian\nCE\nGlobal\nRF\nTrain 100x\nN=9'
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


runs_augmented_histories_df = runs_configs_df[[
    'run_id', 'run_group', 'place_field_loss', 'place_field_values',
    'place_field_normalization', 'place_cell_rf', 'n_grad_steps_per_epoch']].merge(
        runs_histories_df,
        on='run_id',
        how='left')

runs_augmented_histories_df['num_grad_steps'] = \
    runs_augmented_histories_df['n_grad_steps_per_epoch'] * runs_augmented_histories_df['_step']

plot_max_grid_score_vs_num_grad_steps(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir,
)

plot_max_grid_score_vs_num_grad_steps_by_place_cell_rf(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir)

plot_loss_vs_num_grad_steps(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir)

plot_loss_vs_num_grad_steps_by_place_cell_rf(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir)

plot_pos_decoding_error_vs_num_grad_steps(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir)

plot_pos_decoding_error_vs_num_grad_steps_by_place_cell_rf(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir)

print('Finished!')
