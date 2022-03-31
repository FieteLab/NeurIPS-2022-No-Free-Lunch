import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 20  # was previously 22
sns.set_style("whitegrid")


def plot_max_grid_score_vs_run_group_given_low_pos_decoding_err(
        runs_performance_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold: float = 5.):
    plt.close()
    runs_performance_df[f'pos_decoding_err_below_{low_pos_decoding_err_threshold}'] = \
        runs_performance_df['pos_decoding_err'] < low_pos_decoding_err_threshold

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.stripplot(y="max_grid_score_d=60_n=256",
                  x='run_group',
                  data=runs_performance_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(
        f'Max Grid Score | Pos Err < {low_pos_decoding_err_threshold} cm')
    ax.set_xlabel('Group')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="max_grid_score_d=90_n=256",
                  x='run_group',
                  data=runs_performance_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(None)
    ax.set_xlabel('Group')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'max_grid_score_vs_run_group_given_low_pos_decoding_err.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_percent_have_grid_cells_vs_run_group_given_low_pos_decoding_err(
        runs_performance_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold: float = 5.,
        grid_score_d60_threshold: float = 1.0,
        grid_score_d90_threshold: float = 1.5):
    plt.close()
    runs_performance_df[f'pos_decoding_err_below_{low_pos_decoding_err_threshold}'] = \
        runs_performance_df['pos_decoding_err'] < low_pos_decoding_err_threshold

    runs_performance_df[f'has_grid_d60'] \
        = runs_performance_df['max_grid_score_d=60_n=256'] > grid_score_d60_threshold

    runs_performance_df[f'has_grid_d90'] \
        = runs_performance_df['max_grid_score_d=90_n=256'] > grid_score_d90_threshold

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8),
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
    ax.set_xlabel('Group')

    ax = axes[1]
    sns.barplot(y="has_grid_d90",
                x='run_group',
                data=runs_performance_df,
                ax=ax)
    ax.set_ylim(0., 1.)
    ax.set_title(r'$90^{\circ}$')
    ax.set_ylabel(f'Frac Runs : Max Grid Score > {grid_score_d90_threshold} | Pos Err < {low_pos_decoding_err_threshold}')
    ax.set_xlabel('Group')

    plt.savefig(os.path.join(plot_dir,
                             f'percent_have_grid_cells_vs_run_group_given_low_pos_decoding_err.png'),
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

    sns.barplot(x="run_group",
                y=f'pos_decoding_err_below_{low_pos_decoding_err_threshold}',
                data=runs_performance_df)
    plt.xlabel('Group')
    plt.ylabel(f'Frac Runs : Pos Error < {low_pos_decoding_err_threshold} cm')
    plt.ylim(0., 1.)
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
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8),
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
    ax.set_ylabel('Pos Decoding Err')
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


def plot_pos_decoding_err_vs_run_group(
        runs_performance_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    sns.stripplot(x="run_group",
                  y="pos_decoding_err",
                  data=runs_performance_df,
                  size=2)
    plt.ylabel('Pos Decoding Err (cm)')
    plt.xlabel('Group')
    plt.savefig(os.path.join(plot_dir,
                             f'pos_decoding_err_vs_run_group.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
