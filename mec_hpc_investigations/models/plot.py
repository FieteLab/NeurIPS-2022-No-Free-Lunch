import matplotlib.pyplot as plt
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


def plot_pos_decoding_error_vs_num_grad_steps(
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


def plot_pos_decoding_error_vs_num_grad_steps_by_place_cell_rf(
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
