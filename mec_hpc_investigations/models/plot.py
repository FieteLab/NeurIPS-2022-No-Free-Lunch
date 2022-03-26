import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 20  # was previously 22
sns.set_style("whitegrid")


def plot_percent_pos_decoding_err_vs_max_grid_score_by_run_group(
        runs_performance_df: pd.DataFrame,
        plot_dir: str):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    ax = axes[0]
    sns.scatterplot(y="pos_decoding_err",
                    x='max_grid_score_d=60_n=256',
                    data=runs_performance_df,
                    hue='run_group',
                    ax=ax,
                    s=2)
    ax.set_ylabel('Pos Decoding Err')
    ax.set_xlabel(r'Max Grid Score ($60^{\circ}$)')

    ax = axes[1]
    sns.scatterplot(y="pos_decoding_err",
                    x='max_grid_score_d=90_n=256',
                    data=runs_performance_df,
                    hue='run_group',
                    ax=ax,
                    s=2)
    ax.set_xlabel(r'Max Grid Score ($60^{\circ}$)')
    ax.set_title(r'$90^{\circ}$')

    plt.savefig(os.path.join(plot_dir,
                             f'percent_pos_decoding_err_vs_max_grid_score.pdf'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_percent_pos_decoding_err_vs_run_group(
        runs_performance_df: pd.DataFrame,
        plot_dir: str):
    sns.barplot(x="run_group",
                y="pos_decoding_err",
                data=runs_performance_df)
    plt.xlabel('Models')
    plt.ylim(0., 1.)
    plt.savefig(os.path.join(plot_dir,
                             f'percent_pos_decoding_err_vs_run_group.pdf'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_percent_pos_decoding_err_below_threshold_vs_run_group(
        runs_performance_df: pd.DataFrame,
        plot_dir: str,
        threshold: float = 5.):
    runs_performance_df[f'pos_decoding_err_below_{threshold}'] = \
        runs_performance_df['pos_decoding_err'] < threshold

    sns.barplot(x="run_group",
                y=f'pos_decoding_err_below_{threshold}',
                data=runs_performance_df)
    plt.xlabel('Group')
    plt.ylabel(f'Pos Decode Error < {threshold}')
    plt.ylim(0., 1.)
    plt.savefig(os.path.join(plot_dir,
                             f'percent_pos_decoding_err_below_threshold_vs_run_group.pdf'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
