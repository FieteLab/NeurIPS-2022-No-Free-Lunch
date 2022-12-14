import os
import shutil

import matplotlib.pyplot as plt

from mec_hpc_investigations.models.analyze import *
from mec_hpc_investigations.models.plot import *

# Declare variables.
notebook_dir = 'notebooks/10_dog_multiple_fields'
data_dir = os.path.join(notebook_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
results_dir = os.path.join(notebook_dir, 'results')
if os.path.exists(results_dir) and os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)

low_pos_decoding_err_threshold_in_cm = 10.
grid_score_d60_threshold = 0.8
grid_score_d90_threshold = 1.5
sweep_ids = [
    'rbrvuf2g',  # Part 1 of sweep
    'wnmp7nx0',  # Part 2 of sweep
    '56legweh',  # Part 3 of sweep
    'lwalddwy',  # Part 4 of sweep
]

runs_configs_df = download_wandb_project_runs_configs(
    wandb_project_path='mec-hpc-investigations',
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    finished_only=True,
    refresh=False)


# Add 1 to Poissons because we used 1+Poisson number of fields
def add_one_to_poissons(row: pd.Series):
    if isinstance(row['n_place_fields_per_cell'], str) and row['n_place_fields_per_cell'].startswith('Poisson'):
        return '1 + ' + row['n_place_fields_per_cell'].replace(' ', '')
    else:
        return str(row['n_place_fields_per_cell'])


runs_configs_df['n_place_fields_per_cell'] = runs_configs_df.apply(
    add_one_to_poissons,
    axis=1)

joblib_files_data_by_run_id_dict = load_runs_joblib_files(
    run_ids=list(runs_configs_df['run_id'].unique()),
    include_additional_data=True)

overwrite_runs_configs_df_values_with_joblib_data(
    runs_configs_df=runs_configs_df,
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)

plot_percent_runs_with_low_pos_decoding_err_pie(
    runs_configs_df=runs_configs_df,
    plot_dir=results_dir,
    low_pos_decoding_err_threshold_in_cm=low_pos_decoding_err_threshold_in_cm)

# Keep only networks that achieved low position decoding error.
low_pos_decoding_indices = runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm
frac_low_pos_decoding_err = low_pos_decoding_indices.mean()
print(f'Frac Low Pos Decoding Err Runs: {frac_low_pos_decoding_err}')
runs_configs_df = runs_configs_df[low_pos_decoding_indices]

neurons_data_by_run_id_df = convert_joblib_files_data_to_neurons_data_df(
    joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict)


plot_percent_runs_with_grid_cells_pie(
    runs_configs_with_scores_max_df,
                                          plot_dir: str)

augmented_neurons_data_by_run_id_df = runs_configs_df[[
    'run_id', 'n_place_fields_per_cell']].merge(
    neurons_data_by_run_id_df,
    on='run_id',
    how='left')

# plot_rate_maps_examples_hexagons_by_score_range(
#     neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df[
#         augmented_neurons_data_by_run_id_df['n_place_fields_per_cell'] == '1 + Poisson(3.0)'],
#     joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
#     plot_dir=results_dir)

# plot_rate_maps_by_run_id(
#     neurons_data_by_run_id_df=neurons_data_by_run_id_df,
#     joblib_files_data_by_run_id_dict=joblib_files_data_by_run_id_dict,
#     plot_dir=results_dir)

percent_neurons_score60_above_threshold_by_run_id_df = compute_percent_neurons_score60_above_threshold_by_run_id_df(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df)

# Plot fraction of runs with grid cell
percent = 1.
percent_neurons_score60_above_threshold_by_run_id_df['has_grid_cells'] = \
    percent_neurons_score60_above_threshold_by_run_id_df['Percent'] > percent

tmp = percent_neurons_score60_above_threshold_by_run_id_df.merge(
    runs_configs_df[['run_id', 'n_place_fields_per_cell']],
    on='run_id',
    how='left'
)

plt.close()
threshold = 1.0
sns.barplot(
    data=tmp[tmp['Grid Score Threshold'] == threshold],
    x='n_place_fields_per_cell',
    y='has_grid_cells',
    estimator=np.sum,
    ci=None)
plt.xlabel('')
plt.ylabel(f'# Runs with Grid Cells\n{percent}% Units > {threshold}')
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
plt.tight_layout()
plt.show()

plot_grid_scores_histograms_by_n_place_fields_per_cell(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_scores_kdes_by_n_place_fields_per_cell(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_periods_histograms_by_n_place_fields_per_cell(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)

plot_grid_periods_kde_by_n_place_fields_per_cell(
    augmented_neurons_data_by_run_id_df=augmented_neurons_data_by_run_id_df,
    plot_dir=results_dir)


# max_grid_scores_by_run_id_df = augmented_neurons_data_by_run_id_df.groupby('run_id').agg(
#     score_60_by_neuron_max=('score_60_by_neuron', 'max'),
#     score_90_by_neuron_max=('score_90_by_neuron', 'max'),
#     n_place_fields_per_cell=('n_place_fields_per_cell', 'first')).reset_index()

print('Finished 10_dog_multiple_fields/10_dog_multiple_fields.py!')
