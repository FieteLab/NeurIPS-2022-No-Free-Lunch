import os

from mec_hpc_investigations.models.analyze import compute_minima_performance_metrics_from_runs_histories, \
    download_wandb_project_runs_configs, download_wandb_project_runs_histories
from mec_hpc_investigations.models.plot import plot_percent_pos_decoding_err_vs_run_group, \
    plot_percent_pos_decoding_err_below_threshold_vs_run_group


plot_dir = 'notebooks/00_broad_sweep_results/results/'
os.makedirs(plot_dir, exist_ok=True)

sweep_ids = [
    '5bpvzhfh',  # Position task loss
]


runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    sweep_ids=sweep_ids,
    finished_only=True)

runs_configs_df['run_group'] = runs_configs_df.apply(
    lambda row: f"{row['place_field_loss']}\n{row['place_field_values']}\n{row['place_field_normalization']}",
    axis=1)

runs_histories_df = download_wandb_project_runs_histories(
    wandb_project_path='mec-hpc-investigations',
    sweep_ids=sweep_ids)

minima_performance_metrics = compute_minima_performance_metrics_from_runs_histories(
    runs_histories_df=runs_histories_df,
)


runs_performance_df = runs_configs_df[[
    'run_id', 'run_group', 'place_field_loss', 'place_field_values', 'place_field_normalization']].merge(
        minima_performance_metrics,
        on='run_id',
        how='left')


plot_percent_pos_decoding_err_vs_run_group(
    runs_performance_df=runs_performance_df,
    plot_dir=plot_dir)

plot_percent_pos_decoding_err_below_threshold_vs_run_group(
    runs_performance_df=runs_performance_df,
    plot_dir=plot_dir)

print(1)
