import os

from mec_hpc_investigations.models.analyze import compute_minima_performance_metrics_from_runs_histories, \
    download_wandb_project_runs_configs, download_wandb_project_runs_histories
from mec_hpc_investigations.models.plot import *

# Declare variables.
notebook_dir = 'notebooks/06_good_grid_cells_optimizer'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 6.
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.5
sweep_ids = [
    'r83jf81o',  # 20: good grid cells
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True)

# Only take rf = 0.12m
runs_configs_df = runs_configs_df[runs_configs_df['place_cell_rf'] == 0.12]

# def add_human_readable_run_id(row: pd.Series):
#     if row['Sweep'] == '05ljtf0t':
#         run_group = 'CE\nDoG\nGlobal\nOthers\nN=72'
#     else:
#         raise ValueError
#     return run_group
#
#
# runs_configs_df['run_group'] = runs_configs_df.apply(
#     add_human_readable_run_id,
#     axis=1)

# Keep only networks that achieved low position decoding error.
runs_configs_df = runs_configs_df[
    runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold]

runs_histories_df = download_wandb_project_runs_histories(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids)

runs_augmented_histories_df = runs_configs_df[[
    'run_id', 'optimizer']].merge(
    runs_histories_df,
    on='run_id',
    how='left')

plot_max_grid_score_vs_num_grad_steps_by_optimizer(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir,
    grid_score_d60_threshold=grid_score_d60_threshold,
    grid_score_d90_threshold=grid_score_d90_threshold,
)

plot_loss_vs_num_grad_steps_by_optimizer(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir)

plot_pos_decoding_err_vs_num_grad_steps_by_optimizer(
    runs_augmented_histories_df=runs_augmented_histories_df,
    plot_dir=results_dir)


print('Finished!')
