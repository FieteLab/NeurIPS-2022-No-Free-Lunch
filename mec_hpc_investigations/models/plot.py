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
#                     col='run_group',
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

def plot_grid_score_vs_optimizer(augmented_neurons_data_by_run_id_df: pd.DataFrame,
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
    ax.set_ylabel(
        f'Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.boxenplot(y="score_90_by_neuron",
                  x='optimizer',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax,
                  # size=2,
                  )
    ax.set_ylabel(None)
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_vs_optimizer.png'),
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
    ax.set_ylabel(
        f'Max Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="score_90_by_neuron_max",
                  x='optimizer',
                  data=max_grid_scores_by_run_id_df,
                  ax=ax,
                  # size=2,
                  )
    ax.set_ylabel(None)
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_max_vs_optimizer.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_score_histograms_by_human_readable_run_id(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str):
    # ncols = 4
    # nrows = int(np.ceil(len(runs_augmented_histories_df['run_id'].unique()) / ncols))
    #
    # fig, axes = plt.subplots(
    #     nrows=nrows,
    #     ncols=ncols,
    #     figsize=(ncols * 4, nrows * 4),
    #     sharex=True,
    #     sharey=True,
    # )

    for idx, (human_readable_run_id, run_id_histories_df) \
            in enumerate(runs_augmented_histories_df.groupby('human_readable_run_id')):
        # ax = axes[idx // ncols, idx % ncols]
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
        axes[0].set_ylabel('Count')
        for ax_idx, d in enumerate(['60', '90']):
            ax = axes[ax_idx]
            # The Histogram field is sometimes dictionaries, sometimes strings.
            # When strings, the string contains single quotes. We need to replace these
            # with double quotes for JSON to work.
            last_histogram_dict = run_id_histories_df.iloc[-1][f'grid_score_histogram_d={d}_n=256']
            if isinstance(last_histogram_dict, str):
                last_histogram_dict = last_histogram_dict.replace("\'", "\"")
                last_histogram_dict = json.loads(last_histogram_dict)
            last_histogram_dict['values'] = np.array(last_histogram_dict['values'])
            last_histogram_dict['bins'] = np.array(last_histogram_dict['bins'])
            ax.bar(
                (last_histogram_dict['bins'][:-1] + last_histogram_dict['bins'][1:]) / 2,
                last_histogram_dict['values'],
                width=last_histogram_dict['bins'][:-1] - last_histogram_dict['bins'][1:])
            ax.set_title(d + r'$^{\circ}$')
            ax.set_xlabel('Grid Score')

        # ax.set_title(human_readable_run_id)
        # fig.suptitle(human_readable_run_id)

        plt.savefig(os.path.join(plot_dir,
                                 f'grid_score_histograms_run=_{human_readable_run_id}.png'),
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


def plot_max_grid_score_given_low_pos_decoding_err_vs_run_group(
        runs_performance_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold: float = 5.):
    plt.close()
    runs_performance_df[f'pos_decoding_err_below_{low_pos_decoding_err_threshold}'] = \
        runs_performance_df['pos_decoding_err'] < low_pos_decoding_err_threshold

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.stripplot(y="max_grid_score_d=60_n=256",
                  x='run_group',
                  data=runs_performance_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(
        f'Max Grid Score | Pos Err < {low_pos_decoding_err_threshold} cm')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="max_grid_score_d=90_n=256",
                  x='run_group',
                  data=runs_performance_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(None)
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'max_grid_score_given_low_pos_decoding_err_vs_run_group.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_max_grid_score_vs_activation(runs_performance_df: pd.DataFrame,
                                      plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.stripplot(y="max_grid_score_d=60_n=256",
                  x='activation',
                  data=runs_performance_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(
        f'Max Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="max_grid_score_d=90_n=256",
                  x='activation',
                  data=runs_performance_df,
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


def plot_max_grid_score_90_vs_max_grid_score_60_by_activation(
        runs_performance_df: pd.DataFrame,
        plot_dir: str,
        grid_score_d60_threshold: float,
        grid_score_d90_threshold: float):
    plt.close()
    sns.scatterplot(
        data=runs_performance_df,
        x='max_grid_score_d=60_n=256',
        y='max_grid_score_d=90_n=256',
        hue='activation',
    )
    plt.hlines(grid_score_d90_threshold, 0., 2., colors='r')
    plt.vlines(grid_score_d60_threshold, 0., 2., colors='r')
    plt.xlim(0., 2.)
    plt.ylim(0., 2.)
    plt.legend(loc='lower left')

    plt.xlabel(r'Max $60^{\circ}$ Score')
    plt.ylabel(r'Max $90^{\circ}$ Score')
    plt.savefig(os.path.join(plot_dir,
                             f'max_grid_score_90_vs_max_grid_score_60_by_activation.png'),
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


def plot_percent_have_grid_cells_given_low_pos_decoding_err_vs_run_group(
        runs_performance_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold: float = 5.,
        grid_score_d60_threshold: float = 1.2,
        grid_score_d90_threshold: float = 1.4):
    plt.close()
    runs_performance_df[f'pos_decoding_err_below_{low_pos_decoding_err_threshold}'] = \
        runs_performance_df['pos_decoding_err'] < low_pos_decoding_err_threshold

    runs_performance_df[f'has_grid_d60'] \
        = runs_performance_df['max_grid_score_d=60_n=256'] > grid_score_d60_threshold

    runs_performance_df[f'has_grid_d90'] \
        = runs_performance_df['max_grid_score_d=90_n=256'] > grid_score_d90_threshold

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.barplot(y="has_grid_d60",
                x='run_group',
                data=runs_performance_df,
                ax=ax)
    ax.set_ylim(0., 1.)
    ax.set_title(r'$60^{\circ}$')
    ax.set_ylabel(
        f'Frac Runs : Max Grid Score > {grid_score_d60_threshold} | Pos Err < {low_pos_decoding_err_threshold} cm')
    ax.set_xlabel('')

    ax = axes[1]
    sns.barplot(y="has_grid_d90",
                x='run_group',
                data=runs_performance_df,
                ax=ax)
    ax.set_ylim(0., 1.)
    ax.set_title(r'$90^{\circ}$')
    ax.set_ylabel(
        f'Frac Runs : Max Grid Score > {grid_score_d90_threshold} | Pos Err < {low_pos_decoding_err_threshold}')
    ax.set_xlabel('')

    plt.savefig(os.path.join(plot_dir,
                             f'percent_have_grid_cells_given_low_pos_decoding_err_vs_run_group.png'),
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


def plot_percent_low_decoding_err_vs_run_group(
        runs_performance_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold: float = 5.):
    plt.close()
    runs_performance_df[f'pos_decoding_err_below_{low_pos_decoding_err_threshold}'] = \
        runs_performance_df['pos_decoding_err'] < low_pos_decoding_err_threshold

    fig, ax = plt.subplots(figsize=(24, 8))
    sns.barplot(x="run_group",
                y=f'pos_decoding_err_below_{low_pos_decoding_err_threshold}',
                data=runs_performance_df,
                ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(f'Frac Runs : Pos Error < {low_pos_decoding_err_threshold} cm')
    ax.set_ylim(0., 1.)
    plt.savefig(os.path.join(plot_dir,
                             f'percent_low_decoding_err_vs_run_group.png'),
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


def plot_pos_decoding_err_vs_max_grid_score_by_run_group(
        runs_performance_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 8),
                             sharey=True, sharex=True)
    ymin = runs_performance_df['pos_decoding_err'].min()
    ymax = runs_performance_df['pos_decoding_err'].max()
    ax = axes[0]
    sns.scatterplot(y="pos_decoding_err",
                    x='max_grid_score_d=60_n=256',
                    data=runs_performance_df,
                    hue='run_group',
                    ax=ax,
                    s=10)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Pos Decoding Err (cm)')
    ax.set_xlabel(r'Max Grid Score')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.scatterplot(y="pos_decoding_err",
                    x='max_grid_score_d=90_n=256',
                    data=runs_performance_df,
                    hue='run_group',
                    ax=ax,
                    s=10)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(None)  # Share Y-Label with subplot to left.
    ax.set_xlabel(r'Max Grid Score')
    ax.set_title(r'$90^{\circ}$')

    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_err_vs_max_grid_score_by_run_group.png'),
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


def plot_pos_decoding_err_vs_run_group(
        runs_performance_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    fig, ax = plt.subplots(figsize=(24, 8))
    sns.stripplot(x="run_group",
                  y="pos_decoding_err",
                  data=runs_performance_df,
                  size=4,
                  ax=ax)
    ax.set_ylim(1., 100.)
    ax.set_ylabel('Pos Decoding Err (cm)')
    ax.set_xlabel('')
    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_err_vs_run_group.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
