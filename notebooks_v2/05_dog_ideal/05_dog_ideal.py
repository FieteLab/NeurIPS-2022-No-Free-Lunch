import os

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare paths.
notebook_dir = 'notebooks_v2/05_dog_ideal'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 6.
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.5
sweep_ids = [
    'gvwcljra',  # DoG with ideal conditions (rf=0.12 or 0.20, adam or RMSprop)
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True,
    refresh=False)

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()))

overwrite_runs_configs_df_values_with_joblib_data(
    runs_configs_df=runs_configs_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

plot_percent_runs_with_low_pos_decoding_err_pie(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir)

# Keep only networks that achieved low position decoding error.
low_pos_decoding_indices = runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm
frac_low_pos_decoding_err = low_pos_decoding_indices.mean()
print(f'Frac Low Pos Decoding Err Runs: {frac_low_pos_decoding_err}')
runs_configs_df = runs_configs_df[low_pos_decoding_indices]

neurons_data_by_run_id_df = convert_joblib_files_data_to_neurons_data_df(
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

plot_rate_maps_examples_hexagons_by_score_range(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
    plot_dir=results_dir)

plot_rate_maps_examples_squares_by_score_range(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
    plot_dir=results_dir)

augmented_neurons_data_by_run_id_df = runs_configs_df[[
    'run_id', 'place_cell_rf', 'optimizer']].merge(
    neurons_data_by_run_id_df,
    on='run_id',
    how='left')

plot_grid_scores_histograms_by_run_id(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_periods_histograms_by_run_id(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    plot_dir=results_dir)

max_grid_scores_by_run_id_df = augmented_neurons_data_by_run_id_df.groupby('run_id').agg(
    score_60_by_neuron_max=('score_60_by_neuron', 'max'),
    score_90_by_neuron_max=('score_90_by_neuron', 'max'),
    place_cell_rf=('place_cell_rf', 'first')).reset_index()

runs_configs_with_scores_max_df = runs_configs_df.merge(
    max_grid_scores_by_run_id_df,
    on='run_id',
    how='left')

plot_percent_runs_with_grid_cells_pie(
    runs_configs_with_scores_max_df=runs_configs_with_scores_max_df,
    plot_dir=results_dir,)

plot_grid_score_max_as_dots_vs_place_cell_rf(
    max_grid_scores_by_run_id_df=max_grid_scores_by_run_id_df,
    plot_dir=results_dir,)

plot_grid_score_max_as_lines_vs_place_cell_rf(
    max_grid_scores_by_run_id_df=max_grid_scores_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kdes(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    plot_dir=results_dir,)

plot_grid_scores_histogram(
    neurons_data_by_run_id_df=neurons_data_by_run_id_df,
    plot_dir=results_dir)


# # Plot each run's histogram of grid score by number of bins.
# import seaborn as sns
# import matplotlib.pyplot as plt
# bins = np.linspace(-1.0, 2.0, 50)
# for run_id, run_data in joblib_files_data_by_run_id_dict.items():
#     plt.close()
#     plt.hist(run_data['score_60_by_neuron_nbins=20'], bins=bins, label='20', alpha=0.4)
#     plt.hist(run_data['score_60_by_neuron_nbins=32'], bins=bins, label='32', alpha=0.4)
#     plt.hist(run_data['score_60_by_neuron_nbins=44'], bins=bins, label='44', alpha=0.4)
#     plt.title(f'Run ID: {run_id}')
#     plt.xlabel('Grid Score')
#     plt.ylabel('Number of Units')
#     plt.legend()
#     # plt.show()
#     plt.savefig(os.path.join(results_dir,
#                              f'grid_scores_histogram_by_nbins_runid={run_id}.png'),
#                 bbox_inches='tight',
#                 dpi=300)
#     plt.close()

print('Finished 05_dog_ideal/05_dog_ideal.py!')
