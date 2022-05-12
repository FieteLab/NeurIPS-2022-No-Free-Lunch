import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 20  # was previously 22
sns.set_style("whitegrid")


# def plot_pos_decoding_err_over_min_pos_decoding_err_vs_epoch_by_run_id(
#         runs_histories_df: pd.DataFrame,
#         plot_dir: str):
#
#     plt.close()
#
#     g = sns.relplot(data=runs_histories_df,
#                     x="_step",
#                     y='pos_decoding_err_over_min_pos_decoding_err',
#                     hue='run_id',
#                     col='human_readable_sweep',
#                     kind='line',
#                     ci=None,
#                     facet_kws=dict(sharex=True,
#                                    sharey=True,
#                                    margin_titles=True)
#                     )
#     g.set(yscale="log")
#     g.set_ylabels('Pos Decoding Err / Min(Pos Decoding Err)', clear_inner=True)
#     g.set_xlabels('Epoch', clear_inner=True)
#     # g.set_titles(col_template="{col_name}")
#
#     plt.savefig(os.path.join(plot_dir,
#                              f'pos_decoding_err_over_min_pos_decoding_err_vs_epoch_by_run_id.png'),
#                 bbox_inches='tight',
#                 dpi=300)
#     # plt.show()
#     plt.close()


def plot_grid_periods_histograms_by_place_cell_rf(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    bins = np.linspace(0, 100, 101)

    for grid_score_threshold in [0.37, 0.8, 0.85, 1.0, 1.18]:
        # non_nan_period_indices = ~augmented_neurons_data_by_run_id_df['period_per_cell'].isna()
        likely_grid_cell_indices = augmented_neurons_data_by_run_id_df['score_60_by_neuron'] > grid_score_threshold
        sns.histplot(x="period_per_cell",
                     data=augmented_neurons_data_by_run_id_df[likely_grid_cell_indices],
                     hue='place_cell_rf',
                     bins=bins,
                     palette='Spectral_r',
                     # legend='full',
                     # legend='False',
                     # kde=True,
                     )
        # https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot
        # Move the legend off to the right.
        # plt.legend(
        #     bbox_to_anchor=(1, 0.5),  # 1 on the x axis, 0.5 on the y axis
        # )
        xlabel = r'$60^{\circ}$ Grid Period'
        # xlabel += f' (N={(non_nan_period_indices.sum())} out of {len(non_nan_period_indices)})'
        plt.xlabel(xlabel)
        plt.title(f'Grid Score Threshold: {grid_score_threshold}')
        plt.savefig(os.path.join(plot_dir,
                                 f'grid_periods_histograms_by_place_cell_rf_threshold={grid_score_threshold}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_grid_periods_kde_by_place_cell_rf(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    for grid_score_threshold in [0.37, 0.8, 0.85, 1.0, 1.18]:
        likely_grid_cell_indices = augmented_neurons_data_by_run_id_df['score_60_by_neuron'] > grid_score_threshold
        sns.kdeplot(x="period_per_cell",
                    data=augmented_neurons_data_by_run_id_df[likely_grid_cell_indices],
                    hue='place_cell_rf',
                    palette='Spectral_r',
                    fill=True,
                    # legend='full',
                    )
        # Move the legend off to the right.
        # plt.legend(
        #     bbox_to_anchor=(1.2, 0.5),  # 1 on the x axis, 0.5 on the y axis
        # )
        xlabel = r'$60^{\circ}$ Grid Period'
        # xlabel += f' (N={(non_nan_period_indices.sum())} out of {len(non_nan_period_indices)})'
        plt.xlabel(xlabel)
        plt.title(f'Grid Score Threshold: {grid_score_threshold}')
        plt.savefig(os.path.join(plot_dir,
                                 f'grid_periods_kde_by_place_cell_rf_threshold={grid_score_threshold}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_grid_periods_histograms_by_place_cell_rf_by_place_cell_ss(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):

    plt.close()
    bins = np.linspace(0, 100, 101)

    for group, group_df in augmented_neurons_data_by_run_id_df.groupby(['place_cell_rf', 'surround_scale']):
        rf, ss = group
        for grid_score_threshold in [0.37, 0.8, 0.85, 1.0, 1.18]:
            likely_grid_cell_indices = group_df['score_60_by_neuron'] > grid_score_threshold
            sns.histplot(x="period_per_cell",
                         data=group_df[likely_grid_cell_indices],
                         # hue='place_cell_rf',
                         palette='Spectral_r',
                         bins=bins,
                         # legend='full',
                         )
            # Move the legend off to the right.
            # plt.legend(
            #     bbox_to_anchor=(1.2, 0.5),  # 1 on the x axis, 0.5 on the y axis
            # )
            xlabel = r'$60^{\circ}$ Grid Period'
            plt.xlabel(xlabel)
            plt.title(f'RF: {rf}, SS: {ss}, Threshold: {grid_score_threshold}')
            plt.savefig(os.path.join(plot_dir,
                                     f'grid_periods_histograms_by_rf={rf}_ss={ss}_threshold={grid_score_threshold}.png'),
                        bbox_inches='tight',
                        dpi=300)
            # plt.show()
            plt.close()


def plot_grid_periods_kde_by_place_cell_rf_by_place_cell_ss(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    for group, group_df in augmented_neurons_data_by_run_id_df.groupby(['place_cell_rf', 'surround_scale']):

        rf, ss = group

        for grid_score_threshold in [0.37, 0.8, 0.85, 1.0, 1.18]:
            likely_grid_cell_indices = group_df['score_60_by_neuron'] > grid_score_threshold
            sns.kdeplot(x="period_per_cell",
                        data=group_df[likely_grid_cell_indices],
                        # hue='place_cell_rf',
                        palette='Spectral_r',
                        fill=True,
                        # legend='full',
                        )
            # Move the legend off to the right.
            # plt.legend(
            #     bbox_to_anchor=(1.2, 0.5),  # 1 on the x axis, 0.5 on the y axis
            # )
            xlabel = r'$60^{\circ}$ Grid Period'
            plt.xlabel(xlabel)
            plt.title(f'RF: {rf}, SS: {ss}, Threshold: {grid_score_threshold}')
            plt.savefig(os.path.join(plot_dir,
                                     f'grid_periods_kde_by_rf={rf}_ss={ss}_threshold={grid_score_threshold}.png'),
                        bbox_inches='tight',
                        dpi=300)
            # plt.show()
            plt.close()


def plot_grid_periods_histograms_by_run_id(
        neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    bins = np.linspace(0, 50, 51)
    for run_id, neurons_data_df in neurons_data_by_run_id_df.groupby('run_id'):
        non_nan_period_indices = ~neurons_data_df['period_per_cell'].isna()
        sns.histplot(x="period_per_cell",
                     data=neurons_data_df,
                     bins=bins)
        xlabel = r'$60^{\circ}$ Grid Period'
        xlabel += f' (N={(non_nan_period_indices.sum())} out of {len(non_nan_period_indices)})'
        plt.xlabel(xlabel)

        plt.savefig(os.path.join(plot_dir,
                                 f'grid_periods_histograms_run={run_id}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_grid_scores_histograms_by_run_id(
        neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    bins = np.linspace(-1., 1.8, 50)
    for run_id, neurons_data_df in neurons_data_by_run_id_df.groupby('run_id'):
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)

        ax = axes[0]
        sns.histplot(x="score_60_by_neuron",
                     data=neurons_data_df,
                     ax=ax,
                     bins=bins)
        ax.set_xlabel('$60^{\circ}$ Grid Score')

        ax = axes[1]
        sns.histplot(x="score_90_by_neuron",
                     data=neurons_data_df,
                     ax=ax,
                     bins=bins)
        ax.set_xlabel(r'$90^{\circ}$ Grid Score')

        plt.savefig(os.path.join(plot_dir,
                                 f'grid_score_histograms_run={run_id}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_grid_scores_vs_architecture(augmented_neurons_data_by_run_id_df: pd.DataFrame,
                                     plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.boxenplot(y="score_60_by_neuron",
                  x='rnn_type',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax)
    ax.set_ylabel(
        f'Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.boxenplot(y="score_90_by_neuron",
                  x='rnn_type',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax)
    ax.set_ylabel(None)
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_vs_architecture.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_scores_vs_optimizer(augmented_neurons_data_by_run_id_df: pd.DataFrame,
                                  plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.boxenplot(y="score_60_by_neuron",
                  x='optimizer',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax,
                  # size=2,
                  )
    ax.set_ylabel(f'Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.boxenplot(y="score_90_by_neuron",
                  x='optimizer',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax,
                  # size=2,
                  )
    ax.set_ylabel(f'Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_vs_optimizer.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_scores_vs_place_cell_rf(augmented_neurons_data_by_run_id_df: pd.DataFrame,
                                      plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)
    ax = axes[0]
    sns.stripplot(y="score_60_by_neuron",
                  x='place_cell_rf',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax)
    ax.set_ylabel(f'Grid Score')
    ax.set_xlabel(r'$\sigma$')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="score_90_by_neuron",
                  x='place_cell_rf',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax,
                  )
    # ax.set_ylabel(None)
    ax.set_ylabel(f'Grid Score')
    ax.set_xlabel(r'$\sigma$')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_scores_vs_place_cell_rf.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_scores_boxen_vs_place_cell_rf(augmented_neurons_data_by_run_id_df: pd.DataFrame,
                                            plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)
    ax = axes[0]
    sns.boxenplot(y="score_60_by_neuron",
                  x='place_cell_rf',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax)
    ax.set_ylabel(f'Grid Score')
    ax.set_xlabel(r'$\sigma$')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.boxenplot(y="score_90_by_neuron",
                  x='place_cell_rf',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax,
                  )
    # ax.set_ylabel(None)
    ax.set_ylabel(f'Grid Score')
    ax.set_xlabel(r'$\sigma$')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_scores_boxen_vs_place_cell_rf.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_scores_vs_place_cell_rf_by_place_cell_ss(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)
    ax = axes[0]
    sns.stripplot(y="score_60_by_neuron",
                  x='place_cell_rf',
                  hue='surround_scale',
                  dodge=True,  # Otherwise different SS will overlap
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax)
    ax.set_ylabel(f'Grid Scores')
    ax.set_xlabel(r'$\sigma$')
    # https://stackoverflow.com/a/68225877/4570472
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="score_90_by_neuron",
                  x='place_cell_rf',
                  hue='surround_scale',
                  dodge=True,  # Otherwise different SS will overlap
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax)
    # ax.set_ylabel(None)
    ax.set_ylabel(f'Grid Scores')
    ax.set_xlabel(r'$\sigma$')
    # https://stackoverflow.com/a/68225877/4570472
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_scores_vs_place_cell_rf_by_place_cell_ss.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_scores_boxen_vs_place_cell_rf_by_place_cell_ss(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)
    ax = axes[0]
    sns.boxenplot(y="score_60_by_neuron",
                  x='place_cell_rf',
                  hue='surround_scale',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax)
    ax.set_ylabel(f'Grid Scores')
    ax.set_xlabel(r'$\sigma$')
    # https://stackoverflow.com/a/68225877/4570472
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlabel(r'$\sigma$')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.boxenplot(y="score_90_by_neuron",
                  x='place_cell_rf',
                  hue='surround_scale',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax)
    ax.set_ylabel(f'Grid Scores')
    ax.set_xlabel(r'$\sigma$')
    # https://stackoverflow.com/a/68225877/4570472
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_xlabel(r'$\sigma$')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_scores_boxen_vs_place_cell_rf_by_place_cell_ss.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_score_max_vs_architecture(max_grid_scores_by_run_id_df: pd.DataFrame,
                                        plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)
    ax = axes[0]
    sns.stripplot(y="score_60_by_neuron_max",
                  x='rnn_type',
                  data=max_grid_scores_by_run_id_df,
                  ax=ax,
                  )
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="score_90_by_neuron_max",
                  x='rnn_type',
                  data=max_grid_scores_by_run_id_df,
                  ax=ax,
                  )
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_max_vs_architecture.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_score_max_vs_optimizer(max_grid_scores_by_run_id_df: pd.DataFrame,
                                     plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)
    ax = axes[0]
    sns.stripplot(y="score_60_by_neuron_max",
                  x='optimizer',
                  data=max_grid_scores_by_run_id_df,
                  ax=ax,
                  # size=2,
                  )
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="score_90_by_neuron_max",
                  x='optimizer',
                  data=max_grid_scores_by_run_id_df,
                  ax=ax,
                  # size=2,
                  )
    # ax.set_ylabel(None)
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_max_vs_optimizer.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_score_max_as_dots_vs_place_cell_rf(max_grid_scores_by_run_id_df: pd.DataFrame,
                                                 plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)
    ax = axes[0]
    sns.stripplot(y="score_60_by_neuron_max",
                  x='place_cell_rf',
                  data=max_grid_scores_by_run_id_df,
                  ax=ax)
    ax.set_ylabel(f'Max Grid Score (Dot = 1 Run)')
    ax.set_xlabel(r'$\sigma$')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="score_90_by_neuron_max",
                  x='place_cell_rf',
                  data=max_grid_scores_by_run_id_df,
                  ax=ax,
                  )
    # ax.set_ylabel(None)
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel(r'$\sigma$')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_max_as_dots_vs_place_cell_rf.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_score_max_as_lines_vs_place_cell_rf(max_grid_scores_by_run_id_df: pd.DataFrame,
                                                  plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)
    ax = axes[0]
    sns.lineplot(y="score_60_by_neuron_max",
                 x='place_cell_rf',
                 data=max_grid_scores_by_run_id_df,
                 ax=ax)
    ax.set_ylabel(f'Max Grid Score (Avg Across Runs)')
    ax.set_xlabel(r'$\sigma$')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.lineplot(y="score_90_by_neuron_max",
                 x='place_cell_rf',
                 data=max_grid_scores_by_run_id_df,
                 ax=ax,
                 )
    # ax.set_ylabel(None)
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel(r'$\sigma$')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_max_as_lines_vs_place_cell_rf.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_score_max_vs_place_cell_rf_by_place_cell_ss(
        runs_configs_with_scores_max_df: pd.DataFrame,
        plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)
    ax = axes[0]
    sns.barplot(y="score_60_by_neuron_max",
                x='place_cell_rf',
                hue='surround_scale',
                data=runs_configs_with_scores_max_df,
                ax=ax)
    ax.set_ylabel(f'Max Grid Score (Avg Across Runs)')
    ax.set_xlabel(r'$\sigma$')
    # https://stackoverflow.com/a/68225877/4570472
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.barplot(y="score_90_by_neuron_max",
                x='place_cell_rf',
                hue='surround_scale',
                data=runs_configs_with_scores_max_df,
                ax=ax,
                )
    # ax.set_ylabel(None)
    # ax.set_ylabel(f'Max Grid Score')
    ax.set_ylabel(f'Max Grid Scores (Avg Across Runs)')
    ax.set_xlabel(r'$\sigma$')
    # https://stackoverflow.com/a/68225877/4570472
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_max_vs_place_cell_rf_by_place_cell_ss.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_score_max_90_vs_grid_score_max_60_by_architecture(
        max_grid_scores_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    sns.scatterplot(
        data=max_grid_scores_by_run_id_df,
        x='score_60_by_neuron_max',
        y='score_90_by_neuron_max',
        hue='rnn_type',
    )
    # plt.hlines(grid_score_d90_threshold, 0., 2., colors='r')
    # plt.vlines(grid_score_d60_threshold, 0., 2., colors='r')
    plt.xlim(0., 2.)
    plt.ylim(0., 2.)
    # plt.legend(loc='lower left')

    plt.xlabel(r'Max $60^{\circ}$ Score')
    plt.ylabel(r'Max $90^{\circ}$ Score')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_max_90_vs_grid_score_max_60_by_architecture.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_loss_over_min_loss_vs_epoch_by_run_id(
        runs_histories_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    sns.lineplot(
        data=runs_histories_df,
        x='_step',
        y='loss_over_min_loss',
        hue='run_id',
        legend=False,
        linewidth=1.5,
    )

    plt.yscale('log')
    plt.ylabel('Loss / Min(Loss)')
    plt.xlabel('Epoch')

    plt.savefig(os.path.join(plot_dir,
                             f'loss_over_min_loss_vs_epoch_by_run_id.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_loss_vs_num_grad_steps(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    sns.lineplot(y="loss",
                 x='num_grad_steps',
                 data=runs_augmented_histories_df)
    plt.ylabel(f'Loss')
    plt.yscale('log')
    plt.xlabel('Num Grad Steps')

    plt.savefig(os.path.join(plot_dir,
                             f'loss_vs_num_grad_steps.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_loss_vs_num_grad_steps_by_optimizer(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    sns.lineplot(y="loss",
                 x='num_grad_steps',
                 hue='optimizer',
                 data=runs_augmented_histories_df)
    plt.ylabel(f'Loss (Avg Across Runs)')
    plt.yscale('log')
    plt.xlabel('Num Grad Steps')

    plt.savefig(os.path.join(plot_dir,
                             f'loss_vs_num_grad_steps_by_optimizer.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_loss_vs_num_grad_steps_by_place_cell_rf(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    sns.lineplot(y="loss",
                 x='num_grad_steps',
                 hue='place_cell_rf',
                 data=runs_augmented_histories_df,
                 legend="full")
    plt.ylabel(f'Loss')
    plt.yscale('log')
    plt.xlabel('Num Grad Steps')

    plt.savefig(os.path.join(plot_dir,
                             f'loss_vs_num_grad_steps_by_place_cell_rf.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_loss_min_vs_architecture(runs_configs_df: pd.DataFrame,
                                  plot_dir: str):
    sns.stripplot(y="loss",
                  x='rnn_type',
                  data=runs_configs_df,
                  )
    plt.ylabel(f'Loss')
    plt.xlabel('')
    plt.savefig(os.path.join(plot_dir,
                             f'loss_min_vs_architecture.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_loss_min_vs_optimizer(runs_configs_df: pd.DataFrame,
                               plot_dir: str):
    sns.stripplot(y="loss",
                  x='optimizer',
                  data=runs_configs_df,
                  )
    plt.ylabel(f'Loss')
    plt.xlabel('')
    plt.savefig(os.path.join(plot_dir,
                             f'loss_min_vs_optimizer.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_max_grid_score_given_low_pos_decoding_err_vs_human_readable_sweep(
        runs_configs_with_scores_max_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold: float = 5.):
    plt.close()
    runs_configs_with_scores_max_df[f'pos_decoding_err_below_{low_pos_decoding_err_threshold}'] = \
        runs_configs_with_scores_max_df['pos_decoding_err'] < low_pos_decoding_err_threshold

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.stripplot(y="max_grid_score_d=60_n=256",
                  x='human_readable_sweep',
                  data=runs_configs_with_scores_max_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(
        f'Max Grid Score | Pos Err < {low_pos_decoding_err_threshold} cm')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="max_grid_score_d=90_n=256",
                  x='human_readable_sweep',
                  data=runs_configs_with_scores_max_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(None)
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'max_grid_score_given_low_pos_decoding_err_vs_human_readable_sweep.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_max_grid_score_vs_activation(runs_configs_with_scores_max_df: pd.DataFrame,
                                      plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.stripplot(y="max_grid_score_d=60_n=256",
                  x='activation',
                  data=runs_configs_with_scores_max_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(
        f'Max Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="max_grid_score_d=90_n=256",
                  x='activation',
                  data=runs_configs_with_scores_max_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(None)
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'max_grid_score_vs_activation.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_max_grid_score_vs_num_grad_steps(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.lineplot(y="max_grid_score_d=60_n=256",
                 x='num_grad_steps',
                 data=runs_augmented_histories_df,
                 ax=ax,
                 )
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('Num Grad Steps')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.lineplot(y="max_grid_score_d=90_n=256",
                 x='num_grad_steps',
                 data=runs_augmented_histories_df,
                 ax=ax)
    ax.set_ylabel(None)  # Use ylabel from left plot
    ax.set_xlabel('Num Grad Steps')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'max_grid_score_vs_num_grad_steps.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_max_grid_score_vs_num_grad_steps_by_optimizer(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str,
        grid_score_d60_threshold: int = 1.2,
        grid_score_d90_threshold: int = 1.5):
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.lineplot(y="max_grid_score_d=60_n=256",
                 x='num_grad_steps',
                 data=runs_augmented_histories_df,
                 ax=ax,
                 hue='optimizer',
                 estimator='max',
                 ci=None,
                 )
    ax.set_ylabel(f'Max Grid Score (Max Across Runs)')
    ax.set_xlabel('Num Grad Steps')
    ax.set_title(r'$60^{\circ}$')
    ax.axhline(y=grid_score_d60_threshold,
               label='Likely Hexagonal Lattices',
               color='r')

    ax = axes[1]
    sns.lineplot(y="max_grid_score_d=90_n=256",
                 x='num_grad_steps',
                 data=runs_augmented_histories_df,
                 hue='optimizer',
                 estimator='max',
                 ci=None,
                 ax=ax)

    ax.set_ylabel(None)  # Use ylabel from left plot
    ax.set_xlabel('Num Grad Steps')
    ax.set_title(r'$90^{\circ}$')
    ax.axhline(y=grid_score_d90_threshold,
               label='Likely Square Lattices',
               color='r')
    plt.savefig(os.path.join(plot_dir,
                             f'max_grid_score_vs_num_grad_steps_by_optimizer.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_max_grid_score_vs_num_grad_steps_by_place_cell_rf(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.lineplot(y="max_grid_score_d=60_n=256",
                 x='num_grad_steps',
                 hue='place_cell_rf',
                 data=runs_augmented_histories_df,
                 ax=ax,
                 legend="full"
                 )
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('Num Grad Steps')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.lineplot(y="max_grid_score_d=90_n=256",
                 x='num_grad_steps',
                 hue='place_cell_rf',
                 data=runs_augmented_histories_df,
                 legend="full",
                 ax=ax)
    ax.set_ylabel(None)  # Use ylabel from left plot
    ax.set_xlabel('Num Grad Steps')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'max_grid_score_vs_num_grad_steps_by_place_cell_rf.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_max_grid_score_vs_place_cell_rf_by_activation(
        runs_performance_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.stripplot(y="max_grid_score_d=60_n=256",
                  x='place_cell_rf',
                  hue='activation',
                  data=runs_performance_df,
                  ax=ax,
                  size=3)
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('Gaussian ' + r'$\sigma$' + ' (m)')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="max_grid_score_d=90_n=256",
                  x='place_cell_rf',
                  hue='activation',
                  data=runs_performance_df,
                  ax=ax,
                  size=3)
    ax.set_ylabel(None)  # Use ylabel from left plot
    ax.set_xlabel('Gaussian ' + r'$\sigma$')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'max_grid_score_vs_place_cell_rf_by_activation.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_max_grid_score_90_vs_max_grid_score_60_by_activation_and_rnn_type(
        runs_performance_df: pd.DataFrame,
        plot_dir: str,
        grid_score_d60_threshold: float,
        grid_score_d90_threshold: float):
    plt.close()
    unique_rnn_types = runs_performance_df['rnn_type'].unique()
    ncols = len(unique_rnn_types)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(16 * ncols, 8),
                             sharey=True, sharex=True)

    axes[0].set_ylabel(r'Max $90^{\circ}$ Score')
    for ax_idx, unique_rnn_type in enumerate(unique_rnn_types):
        ax = axes[ax_idx]
        sns.scatterplot(
            data=runs_performance_df[runs_performance_df['rnn_type'] == unique_rnn_type],
            x='max_grid_score_d=60_n=256',
            y='max_grid_score_d=90_n=256',
            hue='activation',
            ax=ax,
        )
        ax.hlines(grid_score_d90_threshold, 0., 2., colors='r')
        ax.vlines(grid_score_d60_threshold, 0., 2., colors='r')
        ax.set_xlim(0., 2.)
        ax.set_ylim(0., 2.)
        ax.legend(loc='lower left')
        ax.set_xlabel(r'Max $60^{\circ}$ Score')
        ax.set_title(unique_rnn_type)

    plt.savefig(os.path.join(plot_dir,
                             f'max_grid_score_90_vs_max_grid_score_60_by_activation_and_rnn_type.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_neural_predictivity_vs_participation_ratio_by_architecture_and_activation(
        trained_neural_predictivity_and_ID_df: pd.DataFrame,
        plot_dir: str):
    # g = sns.lmplot(
    #     x='participation_ratio',
    #     y='Trained',
    #     hue='Architecture',
    #     style='Activation',
    #     data=trained_neural_predictivity_and_ID_df,)

    g = sns.scatterplot(
        x='participation_ratio',
        y='Trained',
        hue='Architecture',
        style='Activation',
        data=trained_neural_predictivity_and_ID_df, )
    g.legend(
        bbox_to_anchor=(1, 0.5),  # 1 on the x axis, 0.5 on the y axis
        loc='center left',  # Legend goes center-left of anchor
    )
    plt.xlabel('Participation Ratio')
    plt.ylabel('Neural Predictivity')

    plt.savefig(os.path.join(plot_dir,
                             f'neural_predictivity_vs_participation_ratio_by_architecture_and_activation.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_neural_predictivity_vs_rate_maps_rank_by_architecture_and_activation(
        trained_neural_predictivity_and_ID_df: pd.DataFrame,
        plot_dir: str):
    # g = sns.lmplot(
    #     x='participation_ratio',
    #     y='Trained',
    #     hue='Architecture',
    #     style='Activation',
    #     data=trained_neural_predictivity_and_ID_df,)

    g = sns.scatterplot(
        x='rate_maps_rank',
        y='Trained',
        hue='Architecture',
        style='Activation',
        data=trained_neural_predictivity_and_ID_df, )
    g.legend(
        bbox_to_anchor=(1, 0.5),  # 1 on the x axis, 0.5 on the y axis
        loc='center left',  # Legend goes center-left of anchor
    )
    plt.xlabel('Rate Maps Rank')
    plt.ylabel('Neural Predictivity')

    plt.savefig(os.path.join(plot_dir,
                             f'neural_predictivity_vs_rate_maps_rank_by_architecture_and_activation.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_participation_ratio_by_num_grad_steps(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    sns.lineplot(y="participation_ratio",
                 x='num_grad_steps',
                 data=runs_augmented_histories_df)
    plt.ylabel(f'Participation Ratio')
    plt.xlabel('Num Grad Steps')

    plt.savefig(os.path.join(plot_dir,
                             f'participation_ratio_by_num_grad_steps.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_participation_ratio_vs_architecture_and_activation(
        runs_performance_df: pd.DataFrame,
        plot_dir: str):
    sns.barplot(
        data=runs_performance_df,
        x='rnn_type',
        y='participation_ratio',
        hue='activation',
    )

    plt.xlabel('Architecture')
    plt.ylabel('Participation Ratio')
    plt.savefig(os.path.join(plot_dir,
                             f'participation_ratio_vs_architecture_and_activation.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_percent_grid_cells_vs_place_cell_rf_by_threshold(
        augmented_percent_neurons_score60_above_threshold_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    g = sns.lineplot(
        data=augmented_percent_neurons_score60_above_threshold_by_run_id_df,
        x='place_cell_rf',
        y='Percent',
        hue='Grid Score Threshold',
    )
    g.legend(
        bbox_to_anchor=(1, 0.5),  # 1 on the x axis, 0.5 on the y axis
        loc='center left',  # Legend goes center-left of anchor
    )
    plt.xlabel(r'$\sigma$')
    plt.ylabel('% Grid Cells')

    plt.savefig(os.path.join(plot_dir,
                             f'percent_grid_cells_vs_place_cell_rf_by_threshold.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_percent_grid_cells_vs_place_cell_rf_vs_place_cell_ss_by_threshold(
        augmented_percent_neurons_score60_above_threshold_by_run_id_df: pd.DataFrame,
        plot_dir: str):

    plt.close()
    for threshold, group_df in augmented_percent_neurons_score60_above_threshold_by_run_id_df.groupby('Grid Score Threshold'):
        sns.heatmap(
            pd.pivot_table(group_df, index='surround_scale', columns='place_cell_rf', values='Percent'),
            cmap='Spectral_r',
            vmin=0.,
            vmax=100.,
            linewidths=.5)
        plt.xlabel(r'$\sigma$')
        plt.ylabel(r'$s$')
        plt.title(f'% Grid Cells (Threshold={threshold})')

        plt.savefig(os.path.join(plot_dir,
                                 f'percent_grid_cells_vs_place_cell_rf_vs_place_cell_ss_by_threshold={threshold}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()


def plot_percent_have_grid_cells_given_low_pos_decoding_err_vs_human_readable_sweep(
        runs_configs_with_scores_max_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold: float = 5.,
        grid_score_d60_threshold: float = 1.2,
        grid_score_d90_threshold: float = 1.4):
    plt.close()
    runs_configs_with_scores_max_df[f'pos_decoding_err_below_{low_pos_decoding_err_threshold}'] = \
        runs_configs_with_scores_max_df['pos_decoding_err'] < low_pos_decoding_err_threshold

    runs_configs_with_scores_max_df[f'has_grid_d60'] \
        = runs_configs_with_scores_max_df['max_grid_score_d=60_n=256'] > grid_score_d60_threshold

    runs_configs_with_scores_max_df[f'has_grid_d90'] \
        = runs_configs_with_scores_max_df['max_grid_score_d=90_n=256'] > grid_score_d90_threshold

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.barplot(y="has_grid_d60",
                x='human_readable_sweep',
                data=runs_configs_with_scores_max_df,
                ax=ax)
    ax.set_ylim(0., 1.)
    ax.set_title(r'$60^{\circ}$')
    ax.set_ylabel(
        f'Frac Runs : Max Grid Score > {grid_score_d60_threshold} | Pos Err < {low_pos_decoding_err_threshold} cm')
    ax.set_xlabel('')

    ax = axes[1]
    sns.barplot(y="has_grid_d90",
                x='human_readable_sweep',
                data=runs_configs_with_scores_max_df,
                ax=ax)
    ax.set_ylim(0., 1.)
    ax.set_title(r'$90^{\circ}$')
    ax.set_ylabel(
        f'Frac Runs : Max Grid Score > {grid_score_d90_threshold} | Pos Err < {low_pos_decoding_err_threshold}')
    ax.set_xlabel('')

    plt.savefig(os.path.join(plot_dir,
                             f'percent_have_grid_cells_given_low_pos_decoding_err_vs_human_readable_sweep.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_percent_type_lattice_cells_given_low_pos_decoding_err_vs_activation(
        runs_performance_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold: float = 6.,
        grid_score_d60_threshold: float = 1.2,
        grid_score_d90_threshold: float = 1.4):
    plt.close()
    runs_performance_low_pos_decod_err_df = runs_performance_df[
        runs_performance_df['pos_decoding_err'] < low_pos_decoding_err_threshold].copy()

    runs_performance_low_pos_decod_err_df['has_grid_d60'] \
        = runs_performance_low_pos_decod_err_df['max_grid_score_d=60_n=256'] > grid_score_d60_threshold

    runs_performance_low_pos_decod_err_df['has_grid_d90'] \
        = runs_performance_low_pos_decod_err_df['max_grid_score_d=90_n=256'] > grid_score_d90_threshold

    def compute_lattice_group(row: pd.Series):
        if row['has_grid_d60'] & row['has_grid_d90']:
            lattice_group = 'Both'
        elif row['has_grid_d60'] & ~row['has_grid_d90']:
            lattice_group = r'$60\circ$'
        elif ~row['has_grid_d60'] & row['has_grid_d90']:
            lattice_group = r'$90\circ$'
        elif ~row['has_grid_d60'] & ~row['has_grid_d90']:
            lattice_group = 'Neither'
        else:
            raise ValueError
        return lattice_group

    runs_performance_low_pos_decod_err_df['lattice_group'] = runs_performance_low_pos_decod_err_df.apply(
        compute_lattice_group,
        axis=1)

    sns.histplot(x='lattice_group',
                 hue='activation',
                 stat='density',
                 multiple="dodge",
                 common_norm=False,
                 data=runs_performance_low_pos_decod_err_df)
    plt.xlabel(f'(Probable) Lattices (N={len(runs_performance_low_pos_decod_err_df)})')
    plt.ylim(0., 1.)

    plt.savefig(os.path.join(plot_dir,
                             f'percent_type_lattice_cells_given_low_pos_decoding_err_vs_activation.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_percent_low_pos_decoding_err_pie(runs_configs_df: pd.DataFrame,
                                          plot_dir: str,
                                          low_pos_decoding_err_threshold: float = 6.):
    plt.close()

    pos_decoding_err_below_threshold_col = f'pos_decoding_err_below_{low_pos_decoding_err_threshold}'
    runs_configs_df[pos_decoding_err_below_threshold_col] = \
        runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold

    num_runs_per_category = runs_configs_df.groupby(pos_decoding_err_below_threshold_col)[
        pos_decoding_err_below_threshold_col].count()

    plt.pie(
        x=num_runs_per_category.values,
        labels=num_runs_per_category.index.values,
        colors=['tab:blue' if label == True else 'tab:orange'
                for label in num_runs_per_category.index.values],
        # shadow=True,
        autopct='%.0f%%')
    plt.title('Achieves Low Position Decoding Error')

    plt.savefig(os.path.join(plot_dir, f'percent_low_pos_decoding_err_pie.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_pos_decoding_err_vs_max_grid_score_kde(runs_configs_with_scores_max_df: pd.DataFrame,
                                                plot_dir: str):
    plt.close()

    sns.kdeplot(
        x='score_60_by_neuron_max',
        y='score_90_by_neuron_max',
        # cmap=cmap,
        fill=True,
        clip=(-5, 5),
        levels=15,
        data=runs_configs_with_scores_max_df,
    )

    plt.savefig(os.path.join(plot_dir, f'pos_decoding_err_vs_max_grid_score_kde.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_percent_low_decoding_err_vs_human_readable_sweep(
        runs_configs_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold: float = 5.):
    plt.close()
    runs_configs_df[f'pos_decoding_err_below_{low_pos_decoding_err_threshold}'] = \
        runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold

    fig, ax = plt.subplots(figsize=(24, 8))
    sns.barplot(x="human_readable_sweep",
                y=f'pos_decoding_err_below_{low_pos_decoding_err_threshold}',
                data=runs_configs_df,
                ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(f'Frac Runs : Pos Error < {low_pos_decoding_err_threshold} cm')
    ax.set_ylim(0., 1.)
    plt.savefig(os.path.join(plot_dir,
                             f'percent_low_decoding_err_vs_human_readable_sweep.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_pos_decoding_err_over_min_pos_decoding_err_vs_epoch_by_run_id(
        runs_histories_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    sns.lineplot(
        data=runs_histories_df,
        x='_step',
        y='pos_decoding_err_over_min_pos_decoding_err',
        hue='run_id',
        legend=False,
        linewidth=1.5,
    )

    plt.yscale('log')
    plt.ylabel('Pos Decoding Err / Min(Pos Decoding Err)')
    plt.xlabel('Epoch')

    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_err_over_min_pos_decoding_err_vs_epoch_by_run_id.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_pos_decoding_err_vs_max_grid_score_by_human_readable_sweep(
        runs_configs_with_scores_max_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 8),
                             sharey=True, sharex=True)
    ymin = runs_configs_with_scores_max_df['pos_decoding_err'].min()
    ymax = runs_configs_with_scores_max_df['pos_decoding_err'].max()
    ax = axes[0]
    sns.scatterplot(y="pos_decoding_err",
                    x='max_grid_score_d=60_n=256',
                    data=runs_configs_with_scores_max_df,
                    hue='human_readable_sweep',
                    ax=ax,
                    s=10)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Pos Decoding Err (cm)')
    ax.set_xlabel(r'Max Grid Score')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.scatterplot(y="pos_decoding_err",
                    x='max_grid_score_d=90_n=256',
                    data=runs_configs_with_scores_max_df,
                    hue='human_readable_sweep',
                    ax=ax,
                    s=10)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(None)  # Share Y-Label with subplot to left.
    ax.set_xlabel(r'Max Grid Score')
    ax.set_title(r'$90^{\circ}$')

    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_err_vs_max_grid_score_by_human_readable_sweep.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_pos_decoding_err_vs_num_grad_steps(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    sns.lineplot(y="pos_decoding_err",
                 x='num_grad_steps',
                 data=runs_augmented_histories_df)
    plt.ylabel(f'Pos Decoding Error (cm)')
    plt.yscale('log')
    plt.xlabel('Num Grad Steps')

    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_error_vs_num_grad_steps.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_pos_decoding_err_vs_num_grad_steps_by_optimizer(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    sns.lineplot(y="pos_decoding_err",
                 x='num_grad_steps',
                 hue='optimizer',
                 data=runs_augmented_histories_df)
    plt.ylabel(f'Pos Decoding Error (cm)\n(Avg Across Runs)')
    plt.yscale('log')
    plt.xlabel('Num Grad Steps')

    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_error_vs_num_grad_steps_by_optimizer.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_pos_decoding_err_vs_num_grad_steps_by_place_cell_rf(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    sns.lineplot(y="pos_decoding_err",
                 x='num_grad_steps',
                 hue='place_cell_rf',
                 data=runs_augmented_histories_df,
                 legend="full")
    plt.ylabel(f'Pos Decoding Error (cm)')
    plt.yscale('log')
    plt.xlabel('Num Grad Steps')

    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_error_vs_num_grad_steps_by_place_cell_rf.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_pos_decoding_err_min_vs_architecture(runs_configs_df: pd.DataFrame,
                                              plot_dir: str):
    plt.close()
    sns.stripplot(y="pos_decoding_err",
                  x='rnn_type',
                  data=runs_configs_df,
                  )
    plt.axhline(y=100., color='r', linewidth=5)
    plt.text(x=0, y=55, s='Untrained', color='r')
    plt.ylim(0.1, 100.)
    plt.yscale('log')
    plt.ylabel(f'Pos Decoding Err (cm)')
    plt.xlabel('')
    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_err_min_vs_architecture.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_pos_decoding_err_min_vs_optimizer(runs_configs_df: pd.DataFrame,
                                           plot_dir: str):
    plt.close()
    sns.stripplot(y="pos_decoding_err",
                  x='optimizer',
                  data=runs_configs_df,
                  )
    plt.axhline(y=100., color='r', linewidth=5)
    plt.text(x=0, y=55, s='Untrained', color='r')
    plt.ylim(0.1, 100.)
    plt.yscale('log')
    plt.ylabel(f'Pos Decoding Err (cm)')
    plt.xlabel('')
    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_err_min_vs_optimizer.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_pos_decoding_err_vs_human_readable_sweep(
        runs_configs_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    fig, ax = plt.subplots(figsize=(24, 8))
    sns.stripplot(x="human_readable_sweep",
                  y="pos_decoding_err",
                  data=runs_configs_df,
                  size=4,
                  ax=ax)
    ax.set_ylim(1., 100.)
    ax.set_ylabel('Pos Decoding Err (cm)')
    ax.set_xlabel('')
    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_err_vs_human_readable_sweep.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
