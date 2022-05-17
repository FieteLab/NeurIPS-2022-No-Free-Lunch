import joblib
import numpy as np
import os
import pandas as pd
import scipy.ndimage
import skdim
from typing import Dict, List, Tuple, Union
import wandb


def compute_percent_neurons_score60_above_threshold_by_run_id_df(
        augmented_neurons_data_by_run_id_df: pd.DataFrame) -> pd.DataFrame:

    # https://stackoverflow.com/a/50772444/4570472
    percent_neurons_score60_above_threshold_by_run_id_df = augmented_neurons_data_by_run_id_df.groupby(
        'run_id').agg(
        frac_neurons_with_score_60_above_0p3=('score_60_by_neuron', lambda x: 100*x.gt(0.3).mean()),
        frac_neurons_with_score_60_above_0p8=('score_60_by_neuron', lambda x: 100*x.gt(0.8).mean()),
        frac_neurons_with_score_60_above_1p18=('score_60_by_neuron', lambda x: 100*x.gt(1.18).mean())
    ).reset_index()

    percent_neurons_score60_above_threshold_by_run_id_df.rename(
        columns={
            'frac_neurons_with_score_60_above_0p3': '0.3',
            'frac_neurons_with_score_60_above_0p8': '0.8',
            'frac_neurons_with_score_60_above_1p18': '1.18',
        },
        inplace=True
    )

    percent_neurons_score60_above_threshold_by_run_id_df = pd.melt(
        percent_neurons_score60_above_threshold_by_run_id_df,
        id_vars=['run_id'],
        var_name='Grid Score Threshold',
        value_name='Percent')

    percent_neurons_score60_above_threshold_by_run_id_df['Grid Score Threshold'] = \
        pd.to_numeric(percent_neurons_score60_above_threshold_by_run_id_df['Grid Score Threshold'])

    return percent_neurons_score60_above_threshold_by_run_id_df


def compute_rate_maps_participation_ratio_from_joblib_files_data(
        joblib_files_data_by_run_id_dict: Dict[str, Dict[str, np.ndarray]],
        data_dir: str,
        refresh: bool = False,
        sigma: float = 2.) -> pd.DataFrame:

    rate_maps_participation_ratio_by_run_id_df_path = os.path.join(
        data_dir,
        f'rate_maps_participation_ratio_by_run_id_sigma={sigma}.csv')

    if refresh or not os.path.isfile(rate_maps_participation_ratio_by_run_id_df_path):
        run_ids = []
        rate_maps_participation_ratios = []
        for run_id, joblib_files_data in joblib_files_data_by_run_id_dict.items():
            run_ids.append(run_id)
            rate_maps = joblib_files_data['rate_maps']
            rate_maps[np.isnan(rate_maps)] = 0.
            for idx in range(len(rate_maps)):
                rate_maps[idx] = scipy.ndimage.gaussian_filter(rate_maps[idx], sigma=sigma)
            rate_maps = np.reshape(rate_maps, newshape=(rate_maps.shape[0], -1))

            # Use skdim implementation for trustworthiness.
            rate_maps_pr = skdim.id.lPCA(ver='participation_ratio').fit_transform(
                X=rate_maps)
            rate_maps_participation_ratios.append(rate_maps_pr)
            print(f'Computed participation ratio for run_id={run_id}')

        rate_maps_participation_ratio_by_run_id_df = pd.DataFrame.from_dict({
            'run_id': run_ids,
            'rate_maps_participation_ratio': rate_maps_participation_ratios})

        rate_maps_participation_ratio_by_run_id_df.to_csv(
            rate_maps_participation_ratio_by_run_id_df_path,
            index=False)

    else:
        rate_maps_participation_ratio_by_run_id_df = pd.read_csv(
            rate_maps_participation_ratio_by_run_id_df_path,
            index_col=False)

    return rate_maps_participation_ratio_by_run_id_df


def compute_rate_maps_rank_from_joblib_files_data(
        joblib_files_data_by_run_id_dict: Dict[str, Dict[str, np.ndarray]],
        data_dir: str,
        refresh: bool = False,
        sigma: float = 2.) -> pd.DataFrame:

    ratemap_rank_by_run_id_df_path = os.path.join(data_dir, f'ratemap_rank_by_run_id_sigma={sigma}.csv')

    if refresh or not os.path.isfile(ratemap_rank_by_run_id_df_path):

        run_ids = []
        rate_maps_ranks = []
        for run_id, joblib_files_data in joblib_files_data_by_run_id_dict.items():
            run_ids.append(run_id)
            rate_maps = joblib_files_data['rate_maps']
            rate_maps[np.isnan(rate_maps)] = 0.
            for idx in range(len(rate_maps)):
                rate_maps[idx] = scipy.ndimage.gaussian_filter(rate_maps[idx], sigma=sigma)
            rate_maps = np.reshape(rate_maps, newshape=(rate_maps.shape[0], -1))
            rate_maps_ranks.append(np.linalg.matrix_rank(rate_maps))
            print(f'Computed rank for run_id={run_id}')

        ratemap_rank_by_run_id_df = pd.DataFrame.from_dict({
            'run_id': run_ids,
            'rate_maps_rank': rate_maps_ranks})

        ratemap_rank_by_run_id_df.to_csv(
            ratemap_rank_by_run_id_df_path,
            index=False)

    else:
        ratemap_rank_by_run_id_df = pd.read_csv(
            ratemap_rank_by_run_id_df_path,
            index_col=False)

    return ratemap_rank_by_run_id_df


def convert_joblib_files_data_to_neurons_data_df(
    joblib_files_data_by_run_id_dict: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:

    neurons_data_df_list = []
    for run_id, joblib_files_data in joblib_files_data_by_run_id_dict.items():
        neurons_data_df = pd.DataFrame({
            'neuron_idx': np.arange(len(joblib_files_data['score_60_by_neuron_nbins=44'])),
            'score_60_by_neuron': joblib_files_data['score_60_by_neuron_nbins=44'],
            'score_90_by_neuron': joblib_files_data['score_90_by_neuron_nbins=44'],
            'period_per_cell': joblib_files_data['period_per_cell_nbins=44'],
            'period_err_per_cell': joblib_files_data['period_err_per_cell_nbins=44'],
            'orientations_per_cell': joblib_files_data['orientations_per_cell_nbins=44'],
        })
        neurons_data_df['run_id'] = run_id
        neurons_data_df_list.append(neurons_data_df)

    neurons_data_by_run_id_df = pd.concat(neurons_data_df_list)
    return neurons_data_by_run_id_df


def compute_minima_performance_metrics_from_runs_histories(runs_histories_df: pd.DataFrame):
    minima_performance_metrics = runs_histories_df.groupby(['run_id']).agg({
        'max_grid_score_d=60_n=256': 'max',
        'max_grid_score_d=90_n=256': 'max',
        'pos_decoding_err': 'min',
        'loss': 'min',
        'participation_ratio': 'max',
    })

    # Convert run_id from index to column.
    minima_performance_metrics.reset_index(inplace=True)

    return minima_performance_metrics


def download_wandb_project_runs_configs(wandb_project_path: str,
                                        data_dir: str,
                                        sweep_ids: List[str] = None,
                                        finished_only: bool = True,
                                        refresh: bool = False,
                                        ) -> pd.DataFrame:
    runs_configs_df_path = os.path.join(
        data_dir,
        'sweeps=' + ','.join(sweep_ids) + '_runs_configs.csv')
    if refresh or not os.path.isfile(runs_configs_df_path):

        # Download sweep results
        api = wandb.Api(timeout=60)

        # Project is specified by <entity/project-name>
        if sweep_ids is None:
            runs = api.runs(path=wandb_project_path)
        else:
            runs = []
            for sweep_id in sweep_ids:
                sweep = api.sweep(f'rylan/mec-hpc-investigations/{sweep_id}')
                runs.extend([run for run in sweep.runs])
                # runs.extend(api.runs(path=wandb_project_path,
                #                      filters={"Sweep": sweep_id}))

        sweep_results_list = []
        for run in runs:
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            summary = run.summary._json_dict

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            summary.update(
                {k: v for k, v in run.config.items()
                 if not k.startswith('_')})

            summary.update({'State': run.state,
                            'Sweep': run.sweep.id if run.sweep is not None else None,
                            'run_id': run.id})
            # .name is the human-readable name of the run.
            summary.update({'run_name': run.name})
            sweep_results_list.append(summary)

        runs_configs_df = pd.DataFrame(sweep_results_list)
        runs_configs_df.to_csv(runs_configs_df_path, index=False)
        print(f'Wrote {runs_configs_df_path} to disk.')
    else:
        runs_configs_df = pd.read_csv(runs_configs_df_path)
        print(f'Loaded {runs_configs_df_path} from disk.')

    # Keep only finished runs
    finished_runs = runs_configs_df['State'] == 'finished'
    print(f"% of successfully finished runs: {finished_runs.mean()} ({finished_runs.sum()} / {len(finished_runs)})")

    if finished_only:
        runs_configs_df = runs_configs_df[finished_runs]

        # Check that we don't have an empty data frame.
        assert len(runs_configs_df) > 0

        # Ensure we aren't working with a slice.
        runs_configs_df = runs_configs_df.copy()

    return runs_configs_df


def download_wandb_project_runs_histories(wandb_project_path: str,
                                          data_dir: str,
                                          sweep_ids: List[str] = None,
                                          num_samples: int = 10000,
                                          refresh: bool = False,
                                          keys: List[str] = None,
                                          ) -> pd.DataFrame:
    if keys is None:
        keys = ['max_grid_score_d=60_n=256',
                'max_grid_score_d=90_n=256',
                'pos_decoding_err',
                'participation_ratio',
                'loss',
                'num_grad_steps',
                ]

    runs_histories_df_path = os.path.join(
        data_dir,
        'sweeps=' + ','.join(sweep_ids) + '_runs_histories.csv')
    if refresh or not os.path.isfile(runs_histories_df_path):

        # Download sweep results
        api = wandb.Api(timeout=60)

        # Project is specified by <entity/project-name>
        if sweep_ids is None:
            runs = api.runs(path=wandb_project_path)
        else:
            runs = []
            for sweep_id in sweep_ids:
                sweep = api.sweep(f'rylan/mec-hpc-investigations/{sweep_id}')
                runs.extend([run for run in sweep.runs])
                # runs.extend(api.runs(path=wandb_project_path,
                #                      filters={"Sweep": sweep_id}))

        runs_histories_list = []
        for run in runs:
            run_history_df = run.history(
                samples=num_samples,
                keys=keys)
            if len(run_history_df) == 0:
                continue
            run_history_df['run_id'] = run.id
            runs_histories_list.append(run_history_df)

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        runs_histories_df.sort_values(
            ['run_id', '_step'],
            ascending=True,
            inplace=True)

        runs_histories_df['loss_over_min_loss'] = runs_histories_df.groupby('run_id')['loss'].apply(
            lambda col: col / np.min(col)
        )
        runs_histories_df.reset_index(inplace=True, drop=True)

        runs_histories_df['pos_decoding_err_over_min_pos_decoding_err'] = runs_histories_df.groupby('run_id')[
            'pos_decoding_err'].apply(
            lambda col: col / np.min(col)
        )
        runs_histories_df.reset_index(inplace=True, drop=True)

        runs_histories_df.to_csv(runs_histories_df_path, index=False)
        print(f'Wrote {runs_histories_df_path} to disk')
    else:
        runs_histories_df = pd.read_csv(runs_histories_df_path)
        print(f'Loaded {runs_histories_df_path} from disk.')

    return runs_histories_df


def load_runs_joblib_files(run_ids: List[str],
                           results_dir: str = 'results',
                           ) -> Dict[str, Dict[str, np.ndarray]]:

    joblib_files_data_by_run_id_dict = {run_id: dict() for run_id in run_ids}
    missing_joblib_runs = []
    for run_id in run_ids:
        run_dir = os.path.join(results_dir, run_id)

        try:
            # Extract loss, position decoding error and intrinsic dimensionalities.
            loss_pos_and_dimensionalities_joblib_path = os.path.join(run_dir, 'loss_pos_and_dimensionalities.joblib')
            loss_pos_and_dimensionalities_results = joblib.load(loss_pos_and_dimensionalities_joblib_path)

            # Extract rate maps and scores.
            rate_maps_and_scores_joblib_path = os.path.join(run_dir, 'rate_maps_and_scores.joblib')
            rate_maps_and_scores_results = joblib.load(rate_maps_and_scores_joblib_path)

            # Extract periods, period errors and orientations.
            period_and_orientation_joblib_path = os.path.join(run_dir, 'period_results_path.joblib')
            period_and_orientation_results = joblib.load(period_and_orientation_joblib_path)

            joblib_files_data_by_run_id_dict[run_id] = {
                'loss': loss_pos_and_dimensionalities_results['loss'],
                'pos_decoding_err': loss_pos_and_dimensionalities_results['pos_decoding_err'],
                'participation_ratio': loss_pos_and_dimensionalities_results['participation_ratio'],
                'method_of_moments_ID': loss_pos_and_dimensionalities_results['method_of_moments_ID'],
                'two_NN': loss_pos_and_dimensionalities_results['two_NN'],
                'rate_maps': rate_maps_and_scores_results['rate_maps_nbins=20'],
                'score_60_by_neuron_nbins=20': rate_maps_and_scores_results['score_60_by_neuron_nbins=20'],
                'score_90_by_neuron_nbins=20': rate_maps_and_scores_results['score_90_by_neuron_nbins=20'],
                'period_per_cell_nbins=20': period_and_orientation_results['period_per_cell_nbins=20'],
                'period_err_per_cell_nbins=20': period_and_orientation_results['period_err_per_cell_nbins=20'],
                'orientations_per_cell_nbins=20': period_and_orientation_results['orientations_per_cell_nbins=20'],
                'score_60_by_neuron_nbins=32': rate_maps_and_scores_results['score_60_by_neuron_nbins=32'],
                'score_90_by_neuron_nbins=32': rate_maps_and_scores_results['score_90_by_neuron_nbins=32'],
                'period_per_cell_nbins=32': period_and_orientation_results['period_per_cell_nbins=32'],
                'period_err_per_cell_nbins=32': period_and_orientation_results['period_err_per_cell_nbins=32'],
                'orientations_per_cell_nbins=32': period_and_orientation_results['orientations_per_cell_nbins=32'],
                'score_60_by_neuron_nbins=44': rate_maps_and_scores_results['score_60_by_neuron_nbins=44'],
                'score_90_by_neuron_nbins=44': rate_maps_and_scores_results['score_90_by_neuron_nbins=44'],
                'period_per_cell_nbins=44': period_and_orientation_results['period_per_cell_nbins=44'],
                'period_err_per_cell_nbins=44': period_and_orientation_results['period_err_per_cell_nbins=44'],
                'orientations_per_cell_nbins=44': period_and_orientation_results['orientations_per_cell_nbins=44'],
            }

        except FileNotFoundError:
            missing_joblib_runs.append(run_id)

    if len(missing_joblib_runs) > 0:
        raise FileNotFoundError(f'The following {len(missing_joblib_runs)} runs are missing joblib files:\n{missing_joblib_runs}')

    print('Loaded joblib files\' data')

    return joblib_files_data_by_run_id_dict


def overwrite_runs_configs_df_values_with_joblib_data(
        runs_configs_df: pd.DataFrame,
        joblib_files_data_by_run_id_dict: Dict[str, Dict[str, np.ndarray]]) -> None:

    # Overwrite runs_configs_df's losses and pos decoding errors with post-evaluation
    # values. We do this because the post-training loss and decoding are computed
    # using newer code and longer trajectories.
    keys = ['loss', 'pos_decoding_err', 'participation_ratio', 'two_NN', 'method_of_moments_ID']
    for key in keys:
        runs_configs_df[key] = runs_configs_df.apply(
            lambda row: joblib_files_data_by_run_id_dict[row['run_id']][key],
            axis=1)

    # run_ids = []
    # replacement_pos_decoding_errs = []
    # for key, value in joblib_files_data_by_run_id_dict.items():
    #     run_ids.append(key)
    #     replacement_pos_decoding_errs.append(value['pos_decoding_err'])
    #
    # replacement_df = pd.DataFrame.from_dict({
    #     'run_id': run_ids,
    #     'replacement_pos_decoding_err': replacement_pos_decoding_errs
    # })
    # tmp = runs_configs_df.merge(
    #     replacement_df,
    #     on='run_id',
    #     how='left')
    #
    # tmp[(tmp['replacement_pos_decoding_err'] > 6.) & (tmp['pos_decoding_err'] <= 6.0)]
