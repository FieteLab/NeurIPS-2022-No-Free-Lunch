import json

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.ndimage
import seaborn as sns
from sklearn.neighbors import KernelDensity
from typing import Dict, List

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 20  # was previously 22
sns.set_style("whitegrid")

architecture_color_map = dict(
    RNN='tab:blue',
    LSTM='tab:red',
    GRU='tab:green',
    UGRNN='tab:orange',
    # SRNN='tab:cyan',
)

activation_marker_map = dict(
    relu='o',
    linear='P',
    sigmoid='X',
    tanh='*',
)


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

def plot_grid_periods_histograms_by_n_place_fields_per_cell(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    bins = np.linspace(0, 100, 101)

    for grid_score_threshold in [0.3, 0.8, 1.15]:
        # non_nan_period_indices = ~augmented_neurons_data_by_run_id_df['period_per_cell'].isna()
        likely_grid_cell_indices = augmented_neurons_data_by_run_id_df['score_60_by_neuron'] > grid_score_threshold
        g = sns.histplot(x="period_per_cell",
                         data=augmented_neurons_data_by_run_id_df[likely_grid_cell_indices],
                         hue='n_place_fields_per_cell',
                         bins=bins,
                         palette='Spectral_r',
                         # legend='full',
                         # legend='False',
                         # kde=True,
                         )
        g.legend_.set_title('Num Fields / Place Cell')
        # https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot
        # Move the legend off to the right.
        # plt.legend(
        #     bbox_to_anchor=(1, 0.5),  # 1 on the x axis, 0.5 on the y axis
        # )
        xlabel = r'$60^{\circ}$ Grid Period (cm)'
        # xlabel += f' (N={(non_nan_period_indices.sum())} out of {len(non_nan_period_indices)})'
        plt.xlabel(xlabel)
        plt.title(f'Grid Score Threshold: {grid_score_threshold}')
        plot_path = os.path.join(plot_dir,
                                 f'grid_periods_histograms_by_n_place_fields_per_cell_threshold={grid_score_threshold}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_grid_periods_histograms_by_place_cell_rf(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    bins = np.linspace(0, 100, 101)

    for grid_score_threshold in [0.3, 0.8, 1.15]:
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
        xlabel = r'$60^{\circ}$ Grid Period (cm)'
        # xlabel += f' (N={(non_nan_period_indices.sum())} out of {len(non_nan_period_indices)})'
        plt.xlabel(xlabel)
        plt.title(f'Grid Score Threshold: {grid_score_threshold}')
        plot_path = os.path.join(plot_dir,
                                 f'grid_periods_histograms_by_place_cell_rf_threshold={grid_score_threshold}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_grid_periods_kde_by_n_place_fields_per_cell(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    for grid_score_threshold in [0.3, 0.8, 1.15]:
        likely_grid_cell_indices = augmented_neurons_data_by_run_id_df['score_60_by_neuron'] > grid_score_threshold
        g = sns.kdeplot(x="period_per_cell",
                        data=augmented_neurons_data_by_run_id_df[likely_grid_cell_indices],
                        hue='n_place_fields_per_cell',
                        # palette='Spectral_r',
                        fill=True,
                        # legend='full',
                        )
        # Remove the legend and replace with a color bar.
        # https://stackoverflow.com/questions/62884183/trying-to-add-a-colorbar-to-a-seaborn-scatterplot
        g.legend_.set_title('Num Fields / Place Cell')
        xlabel = r'$60^{\circ}$ Grid Period (cm)'
        # xlabel += f' (N={(non_nan_period_indices.sum())} out of {len(non_nan_period_indices)})'
        plt.xlabel(xlabel)
        plt.title(f'Grid Score Threshold: {grid_score_threshold}')
        plot_path = os.path.join(plot_dir,
                                 f'grid_periods_kde_by_n_place_fields_per_cell_threshold={grid_score_threshold}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_grid_periods_kde_by_place_cell_rf(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    norm = plt.Normalize(
        augmented_neurons_data_by_run_id_df['place_cell_rf'].min(),
        augmented_neurons_data_by_run_id_df['place_cell_rf'].max())
    sm = plt.cm.ScalarMappable(cmap="Spectral_r", norm=norm)
    sm.set_array([])

    for grid_score_threshold in [0.3, 0.8, 1.15]:
        likely_grid_cell_indices = augmented_neurons_data_by_run_id_df['score_60_by_neuron'] > grid_score_threshold
        sns.kdeplot(x="period_per_cell",
                    data=augmented_neurons_data_by_run_id_df[likely_grid_cell_indices],
                    hue='place_cell_rf',
                    palette='Spectral_r',
                    fill=True,
                    # legend='full',
                    )
        # Remove the legend and replace with a color bar.
        # https://stackoverflow.com/questions/62884183/trying-to-add-a-colorbar-to-a-seaborn-scatterplot
        plt.gca().get_legend().remove()
        plt.gca().figure.colorbar(sm, label=r'$\sigma$')
        xlabel = r'$60^{\circ}$ Grid Period (cm)'
        # xlabel += f' (N={(non_nan_period_indices.sum())} out of {len(non_nan_period_indices)})'
        plt.xlabel(xlabel)
        plt.title(f'Grid Score Threshold: {grid_score_threshold}')
        plot_path = os.path.join(plot_dir,
                                 f'grid_periods_kde_by_place_cell_rf_threshold={grid_score_threshold}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_grid_periods_kde_facet_by_place_cell_rf(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    norm = plt.Normalize(
        augmented_neurons_data_by_run_id_df['place_cell_rf'].min(),
        augmented_neurons_data_by_run_id_df['place_cell_rf'].max())
    sm = plt.cm.ScalarMappable(cmap="Spectral_r", norm=norm)
    sm.set_array([])

    for grid_score_threshold in [0.3, 0.8, 1.0, 1.15]:
        likely_grid_cell_indices = augmented_neurons_data_by_run_id_df['score_60_by_neuron'] > grid_score_threshold
        plt.close()
        g = sns.displot(
            data=augmented_neurons_data_by_run_id_df[likely_grid_cell_indices],
            x='period_per_cell',
            col='place_cell_rf',
            hue='place_cell_rf',
            kind='hist',
            kde=True,
            palette='Spectral_r',
            # sharex=True,
            # sharey=True,
        )
        g.set_titles(r"$\sigma={col_name}$")  # use this argument literally
        g.set_xlabels(r'$60^{\circ}$ Grid Period (cm)')
        g.set_ylabels(r'Number of Units')
        plot_path = os.path.join(plot_dir,
                                 f'grid_periods_kde_facet_by_place_cell_rf_threshold={grid_score_threshold}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_grid_periods_mode_vs_place_cell_rf(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    norm = plt.Normalize(
        augmented_neurons_data_by_run_id_df['place_cell_rf'].min(),
        augmented_neurons_data_by_run_id_df['place_cell_rf'].max())
    sm = plt.cm.ScalarMappable(cmap="Spectral_r", norm=norm)
    sm.set_array([])

    # bins = np.linspace(0, 100, 50)
    x_assess_values = np.linspace(0, 150, 1000)[:, np.newaxis]

    # Construct KDE plot
    for grid_score_threshold in [0.3, 0.8, 1.15]:

        plt.close()

        place_cell_rfs, place_cell_rfs_modes = [], []
        for rf in augmented_neurons_data_by_run_id_df['place_cell_rf'].unique():
            # Construct KDE, then extract lines to get mode.
            indices = (augmented_neurons_data_by_run_id_df['place_cell_rf'] == rf) \
                      & (augmented_neurons_data_by_run_id_df['score_60_by_neuron'].values > grid_score_threshold) \
                      & (~pd.isna(augmented_neurons_data_by_run_id_df['period_per_cell']))
            if indices.sum() == 0:
                continue
            kde = KernelDensity(kernel='gaussian').fit(
                augmented_neurons_data_by_run_id_df['period_per_cell'].values[indices][:, np.newaxis])
            log_density_values = kde.score_samples(x_assess_values)
            mode = x_assess_values[np.argmax(log_density_values), 0]
            place_cell_rfs.append(rf)
            place_cell_rfs_modes.append(mode)

        sns.scatterplot(data=pd.DataFrame.from_dict({'kde_mode': place_cell_rfs_modes,
                                                     'place_cell_rf': place_cell_rfs}),
                        x='place_cell_rf',
                        y='kde_mode',
                        s=150,
                        hue='place_cell_rf',
                        palette='Spectral_r')
        # x = np.sort(augmented_neurons_data_by_run_id_df['place_cell_rf'].unique())
        # y = 100 * x / np.sqrt(2 * np.log(2) / 3)
        # plt.plot(x, y)
        # plt.show()
        plt.gca().get_legend().remove()
        plt.gca().figure.colorbar(sm, label=r'$\sigma$')

        plt.ylabel('Grid Period Mode (cm)')
        plt.xlabel(r'$\sigma$ (m)')
        plt.title(f'Grid Score Threshold: {grid_score_threshold}')
        plot_path = os.path.join(plot_dir,
                                 f'grid_periods_mode_vs_place_cell_rf_threshold={grid_score_threshold}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_grid_periods_mode_ratios_vs_index(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    norm = plt.Normalize(
        augmented_neurons_data_by_run_id_df['place_cell_rf'].min(),
        augmented_neurons_data_by_run_id_df['place_cell_rf'].max())
    sm = plt.cm.ScalarMappable(cmap="Spectral_r", norm=norm)
    sm.set_array([])

    # bins = np.linspace(0, 100, 50)
    x_assess_values = np.linspace(0, 150, 1000)[:, np.newaxis]

    # Construct KDE plot
    for grid_score_threshold in [0.3, 0.8, 1.15]:
        plt.close()
        place_cell_rfs, grid_period_modes = [], []
        for rf in sorted(augmented_neurons_data_by_run_id_df['place_cell_rf'].unique()):
            # Construct KDE, then extract lines to get mode.
            indices = (augmented_neurons_data_by_run_id_df['place_cell_rf'] == rf) \
                      & (augmented_neurons_data_by_run_id_df['score_60_by_neuron'].values > grid_score_threshold) \
                      & (~pd.isna(augmented_neurons_data_by_run_id_df['period_per_cell']))
            if indices.sum() == 0:
                continue
            kde = KernelDensity(kernel='gaussian').fit(
                augmented_neurons_data_by_run_id_df['period_per_cell'].values[indices][:, np.newaxis])
            log_density_values = kde.score_samples(x_assess_values)
            mode = x_assess_values[np.argmax(log_density_values), 0]
            place_cell_rfs.append(rf)
            grid_period_modes.append(mode)
        grid_period_modes = np.array(grid_period_modes)
        ratios = np.divide(grid_period_modes[1:], grid_period_modes[:-1])
        g = sns.scatterplot(data=pd.DataFrame.from_dict({'i': 1 + np.arange(len(ratios)),
                                                         'ratios': ratios}),
                            x='i',
                            y='ratios',
                            s=150,
                            # hue='place_cell_rf',
                            palette='Spectral_r')
        g.set(xticks=1 + np.arange(len(ratios)))
        # plt.gca().get_legend().remove()
        # plt.gca().figure.colorbar(sm, label=r'$\sigma$')
        plt.ylabel('Grid Period Mode Ratios')
        plt.xlabel(r'Index')
        plt.title(f'Grid Score Threshold: {grid_score_threshold}')
        plot_path = os.path.join(plot_dir,
                                 f'grid_periods_mode_ratios_vs_index_threshold={grid_score_threshold}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_grid_periods_histograms_by_place_cell_rf_by_place_cell_ss(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    bins = np.linspace(0, 100, 101)

    for group, group_df in augmented_neurons_data_by_run_id_df.groupby(['place_cell_rf', 'surround_scale']):
        rf, ss = group
        for grid_score_threshold in [0.3, 0.8, 1.15]:
            likely_grid_cell_indices = group_df['score_60_by_neuron'] > grid_score_threshold
            sns.histplot(x="period_per_cell",
                         data=group_df[likely_grid_cell_indices],
                         # hue='place_cell_rf',
                         palette='Spectral_r',
                         bins=bins,
                         kde=True,
                         # legend='full',
                         )
            # Move the legend off to the right.
            # plt.legend(
            #     bbox_to_anchor=(1.2, 0.5),  # 1 on the x axis, 0.5 on the y axis
            # )
            plt.xlabel(r'$60^{\circ}$ Grid Period (cm)')
            plt.ylabel('Number of Units')
            plt.title(r'$\sigma=$' + f'{rf}\n' + r'$s=$' + f'{ss}\nGrid Score Threshold: {grid_score_threshold}')
            plot_path = os.path.join(
                plot_dir,
                f"grid_periods_histograms_by_rf={rf.replace(' ', '')}"
                f"_ss={ss.replace(' ', '')}_threshold={grid_score_threshold}.png")
            plt.savefig(plot_path,
                        bbox_inches='tight',
                        dpi=300)
            # plt.show()
            plt.close()
            print(f'Plotted {plot_path}')


def plot_grid_periods_kde_by_place_cell_rf_by_place_cell_ss(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    for group, group_df in augmented_neurons_data_by_run_id_df.groupby(['place_cell_rf', 'surround_scale']):

        rf, ss = group

        for grid_score_threshold in [0.3, 0.8, 1.15]:
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
            xlabel = r'$60^{\circ}$ Grid Period (cm)'
            plt.xlabel(xlabel)
            plt.title(r'$\sigma=$' + f'{rf}\n' + r'$s=$' + f'{ss}\nGrid Score Threshold: {grid_score_threshold}')
            plot_path = os.path.join(
                plot_dir,
                f"grid_periods_kde_by_rf={rf.replace(' ', '')}"
                f"_ss={ss.replace(' ', '')}"
                f"_threshold={grid_score_threshold}.png")
            plt.savefig(plot_path,
                        bbox_inches='tight',
                        dpi=300)
            # plt.show()
            plt.close()
            print(f'Plotted {plot_path}')


def plot_grid_periods_histograms_by_run_id(
        neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    bins = np.linspace(0, 50, 51)
    for run_id, neurons_data_df in neurons_data_by_run_id_df.groupby('run_id'):
        non_nan_period_indices = ~neurons_data_df['period_per_cell'].isna()
        sns.histplot(x="period_per_cell",
                     data=neurons_data_df,
                     bins=bins)
        xlabel = r'$60^{\circ}$ Grid Period (cm)'
        xlabel += f' (N={(non_nan_period_indices.sum())} out of {len(non_nan_period_indices)})'
        plt.xlabel(xlabel)

        plot_path = os.path.join(plot_dir,
                                 f'grid_periods_histograms_run={run_id}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_grid_scores_histogram(
        neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    bins = np.linspace(-1., 1.8, 50)
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)

    ax = axes[0]
    sns.histplot(x="score_60_by_neuron",
                 data=neurons_data_by_run_id_df,
                 ax=ax,
                 bins=bins)
    ax.set_xlabel('$60^{\circ}$ Grid Score')
    ax.set_yscale('log')

    ax = axes[1]
    sns.histplot(x="score_90_by_neuron",
                 data=neurons_data_by_run_id_df,
                 ax=ax,
                 bins=bins)
    ax.set_xlabel(r'$90^{\circ}$ Grid Score')
    ax.set_yscale('log')

    plot_path = os.path.join(plot_dir, f'grid_scores_histogram.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_grid_scores_histograms_by_activation(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    bins = np.linspace(-0.6, 1.4, 75)
    g = sns.histplot(
        data=augmented_neurons_data_by_run_id_df,
        x='score_60_by_neuron',
        bins=bins,
        kde=True,
        hue='activation')
    plt.yscale('log')
    g.legend_.set_title('Activation')
    plt.xlabel('Grid Score')
    plt.ylabel('Number of Units')
    plot_path = os.path.join(plot_dir,
                             f'grid_scores_histograms_by_activation.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_grid_scores_histograms_by_human_readable_sweep(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    bins = np.linspace(-0.6, 1.4, 75)

    plt.close()
    g = sns.histplot(
        data=augmented_neurons_data_by_run_id_df,
        x='score_60_by_neuron',
        bins=bins,
        # kde=True,
        hue='human_readable_sweep')
    g.legend_.set_title('Sweep')
    plt.xlabel('Grid Score')
    plt.ylabel('Number of Units')
    plt.yscale('log')
    plot_path = os.path.join(plot_dir,
                             f'grid_scores_histograms_by_human_readable_sweep.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_grid_scores_histograms_by_n_place_fields_per_cell(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    bins = np.linspace(-0.6, 1.4, 75)

    plt.close()
    g = sns.histplot(
        data=augmented_neurons_data_by_run_id_df,
        x='score_60_by_neuron',
        bins=bins,
        kde=True,
        hue='n_place_fields_per_cell')
    g.legend_.set_title('Num Fields / Place Cell')
    plt.xlabel('Grid Score')
    plt.ylabel('Number of Units')
    plt.yscale('log')
    plt.gca().set_ylim(bottom=1, top=None)
    plot_path = os.path.join(plot_dir,
                             f'grid_scores_histograms_by_n_place_fields_per_cell.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_grid_scores_histograms_by_optimizer(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    bins = np.linspace(-0.6, 1.4, 75)
    g = sns.histplot(
        data=augmented_neurons_data_by_run_id_df,
        x='score_60_by_neuron',
        bins=bins,
        kde=True,
        hue='optimizer')
    g.legend_.set_title('Optimizer')
    # sns.displot(
    #     data=augmented_neurons_data_by_run_id_df,
    #     x='score_60_by_neuron',
    #     bins=bins,
    #     kde=True,
    #     hue='n_place_fields_per_cell',
    #     col='n_place_fields_per_cell'
    # )

    plt.xlabel('Grid Score')
    plt.ylabel('Number of Units')
    plt.yscale('log')
    plot_path = os.path.join(plot_dir,
                             f'grid_scores_histograms_by_optimizer.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_grid_scores_histograms_by_place_field_values(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    bins = np.linspace(-0.6, 1.4, 75)
    g = sns.histplot(
        data=augmented_neurons_data_by_run_id_df,
        x='score_60_by_neuron',
        bins=bins,
        kde=True,
        hue='place_field_values')
    g.legend_.set_title('Target')
    plt.xlabel('Grid Score')
    plt.ylabel('Number of Units')
    plt.gca().set_ylim(bottom=1, top=1000)
    plt.yscale('log')
    plot_path = os.path.join(plot_dir,
                             f'grid_scores_histograms_by_place_field_values.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_grid_scores_histograms_by_place_cell_rf_and_ss_homo_vs_hetero(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    bins = np.linspace(-0.6, 1.4, 75)
    homogeneous_indices = (augmented_neurons_data_by_run_id_df['place_cell_rf'] == '0.12') \
                          & (augmented_neurons_data_by_run_id_df['surround_scale'] == '2')
    heterogeneous_indices = (augmented_neurons_data_by_run_id_df['place_cell_rf'] == 'Uniform( 0.06 , 0.18 )') \
                            & (augmented_neurons_data_by_run_id_df['surround_scale'] == 'Uniform( 1.50 , 2.50 )')
    indices_to_keep = homogeneous_indices | heterogeneous_indices
    subset_df = augmented_neurons_data_by_run_id_df[indices_to_keep]
    subset_df['DoG Params'] = ''
    subset_df['DoG Params'][homogeneous_indices] = r'$\sigma=0.12$' + '\n' + r'$s=2.0$'
    subset_df['DoG Params'][
        heterogeneous_indices] = r'$\sigma \sim$ Unif(0.06, 0.18)' + '\n' + r'$s \sim$ Unif(1.5, 2.5)'
    sns.histplot(
        data=subset_df,
        x='score_60_by_neuron',
        bins=bins,
        kde=True,
        hue='DoG Params'
    )
    plt.xlabel('Grid Score')
    plt.ylabel('Number of Units')
    plt.yscale('log')
    plot_path = os.path.join(plot_dir,
                             f'grid_scores_histograms_by_place_cell_rf_and_ss_homo_vs_hetero.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


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
        ax.set_yscale('log')

        ax = axes[1]
        sns.histplot(x="score_90_by_neuron",
                     data=neurons_data_df,
                     ax=ax,
                     bins=bins)
        ax.set_xlabel(r'$90^{\circ}$ Grid Score')
        ax.set_yscale('log')

        plot_path = os.path.join(plot_dir,
                                 f'grid_score_histograms_run={run_id}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_grid_scores_kde(
        neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    sns.kdeplot(
        data=neurons_data_by_run_id_df,
        x='score_60_by_neuron')
    plt.xlabel('Grid Score')
    plt.ylabel('Density')
    plot_path = os.path.join(plot_dir, f'grid_scores_kde.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_grid_scores_kdes_by_human_readable_sweep(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    # indices_to_keep = (augmented_neurons_data_by_run_id_df['n_place_fields_per_cell'] == '1') \
    #                   | (augmented_neurons_data_by_run_id_df['n_place_fields_per_cell'] == '1 + Poisson( 1.0 )')
    g = sns.kdeplot(
        # data=augmented_neurons_data_by_run_id_df[indices_to_keep],
        data=augmented_neurons_data_by_run_id_df,
        x='score_60_by_neuron',
        common_norm=False,  # Ensure each sweep is normalized separately.
        hue='human_readable_sweep')
    g.legend_.set_title('Sweep')
    plt.xlabel('Grid Score')
    plt.ylabel('Density')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_scores_kdes_by_human_readable_sweep.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_scores_kdes_by_place_cell_rf_and_ss_homo_vs_hetero(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    homogeneous_indices = (augmented_neurons_data_by_run_id_df['place_cell_rf'] == '0.12') \
                          & (augmented_neurons_data_by_run_id_df['surround_scale'] == '2')
    heterogeneous_indices = (augmented_neurons_data_by_run_id_df['place_cell_rf'] == 'Uniform( 0.06 , 0.18 )') \
                            & (augmented_neurons_data_by_run_id_df['surround_scale'] == 'Uniform( 1.50 , 2.50 )')
    indices_to_keep = homogeneous_indices | heterogeneous_indices
    subset_df = augmented_neurons_data_by_run_id_df[indices_to_keep]
    subset_df['DoG Params'] = ''
    subset_df['DoG Params'][homogeneous_indices] = r'$\sigma=0.12$' + '\n' + r'$s=2.0$'
    subset_df['DoG Params'][
        heterogeneous_indices] = r'$\sigma \sim$ Unif(0.06, 0.18)' + '\n' + r'$s \sim$ Unif(1.5, 2.5)'
    g = sns.kdeplot(
        data=subset_df,
        x='score_60_by_neuron',
        hue='DoG Params')
    g.legend_.set_title('')
    plt.xlabel('Grid Score')
    plt.ylabel('Density')

    plot_path = os.path.join(plot_dir,
                             f'grid_scores_kdes_by_place_cell_rf_and_ss_homo_vs_hetero.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_grid_scores_kdes_by_n_place_fields_per_cell(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    # indices_to_keep = (augmented_neurons_data_by_run_id_df['n_place_fields_per_cell'] == '1') \
    #                   | (augmented_neurons_data_by_run_id_df['n_place_fields_per_cell'] == '1 + Poisson( 1.0 )')
    g = sns.kdeplot(
        # data=augmented_neurons_data_by_run_id_df[indices_to_keep],
        data=augmented_neurons_data_by_run_id_df,
        x='score_60_by_neuron',
        hue='n_place_fields_per_cell')
    # g.legend(
    #     bbox_to_anchor=(1, 0.5),  # 1 on the x axis, 0.5 on the y axis
    #     loc='center left',  # Legend goes center-left of anchor
    #     title='Num Fields / Place Cell',
    # )
    g.legend_.set_title('Num Fields / Place Cell')
    plt.xlabel('Grid Score')
    plt.ylabel('Density')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_scores_kdes_by_n_place_fields_per_cell.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_scores_kdes_by_place_field_values(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()

    # indices_to_keep = (augmented_neurons_data_by_run_id_df['n_place_fields_per_cell'] == '1') \
    #                   | (augmented_neurons_data_by_run_id_df['n_place_fields_per_cell'] == '1 + Poisson( 1.0 )')
    g = sns.kdeplot(
        # data=augmented_neurons_data_by_run_id_df[indices_to_keep],
        data=augmented_neurons_data_by_run_id_df,
        x='score_60_by_neuron',
        # kde=True,
        hue='place_field_values')
    g.legend_.set_title('Target')
    plt.xlabel('Grid Score')
    plt.ylabel('Density')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_scores_kdes_by_place_field_values.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()


def plot_grid_scores_vs_activation(augmented_neurons_data_by_run_id_df: pd.DataFrame,
                                   plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.boxenplot(y="score_60_by_neuron",
                  x='activation',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax,
                  # size=2,
                  )
    ax.set_ylabel(f'Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.boxenplot(y="score_90_by_neuron",
                  x='activation',
                  data=augmented_neurons_data_by_run_id_df,
                  ax=ax,
                  # size=2,
                  )
    ax.set_ylabel(f'Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_vs_activation.png'),
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
    plot_path = os.path.join(plot_dir,
                             f'grid_scores_vs_place_cell_rf.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


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
    plot_path = os.path.join(plot_dir,
                             f'grid_scores_boxen_vs_place_cell_rf.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


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


def plot_grid_score_max_vs_activation(max_grid_scores_by_run_id_df: pd.DataFrame,
                                      plot_dir: str):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)
    ax = axes[0]
    sns.stripplot(y="score_60_by_neuron_max",
                  x='activation',
                  data=max_grid_scores_by_run_id_df,
                  ax=ax,
                  # size=2,
                  )
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="score_90_by_neuron_max",
                  x='activation',
                  data=max_grid_scores_by_run_id_df,
                  ax=ax,
                  # size=2,
                  )
    # ax.set_ylabel(None)
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$90^{\circ}$')
    plt.savefig(os.path.join(plot_dir,
                             f'grid_score_max_vs_activation.png'),
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
    plot_path = os.path.join(plot_dir,
                             f'grid_score_max_as_dots_vs_place_cell_rf.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


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
    plot_path = os.path.join(plot_dir,
                             f'grid_score_max_as_lines_vs_place_cell_rf.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


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
        low_pos_decoding_err_threshold_in_cm: float = 6.):
    plt.close()
    runs_configs_with_scores_max_df[f'pos_decoding_err_below_{low_pos_decoding_err_threshold_in_cm}'] = \
        runs_configs_with_scores_max_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.stripplot(y="score_60_by_neuron_max",
                  x='human_readable_sweep',
                  data=runs_configs_with_scores_max_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(
        f'Max Grid Score | Pos Err < {low_pos_decoding_err_threshold_in_cm} cm')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="score_90_by_neuron_max",
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
    sns.stripplot(y="score_60_by_neuron_max",
                  x='activation',
                  data=runs_configs_with_scores_max_df,
                  ax=ax,
                  size=2)
    ax.set_ylabel(
        f'Max Grid Score')
    ax.set_xlabel('')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="score_90_by_neuron_max",
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
    sns.lineplot(y="score_60_by_neuron_max",
                 x='num_grad_steps',
                 data=runs_augmented_histories_df,
                 ax=ax,
                 )
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('Num Grad Steps')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.lineplot(y="score_90_by_neuron_max",
                 x='num_grad_steps',
                 data=runs_augmented_histories_df,
                 ax=ax)
    ax.set_ylabel(None)  # Use ylabel from left plot
    ax.set_xlabel('Num Grad Steps')
    ax.set_title(r'$90^{\circ}$')
    plot_path = os.path.join(plot_dir,
                             f'max_grid_score_vs_num_grad_steps.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_max_grid_score_vs_num_grad_steps_by_optimizer(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str,
        grid_score_d60_threshold: int = 1.2,
        grid_score_d90_threshold: int = 1.5):
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.lineplot(y="score_60_by_neuron_max",
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
    sns.lineplot(y="score_90_by_neuron_max",
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
    plot_path = os.path.join(plot_dir,
                             f'max_grid_score_vs_num_grad_steps_by_optimizer.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_max_grid_score_vs_num_grad_steps_by_place_cell_rf(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.lineplot(y="score_60_by_neuron_max",
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
    sns.lineplot(y="score_90_by_neuron_max",
                 x='num_grad_steps',
                 hue='place_cell_rf',
                 data=runs_augmented_histories_df,
                 legend="full",
                 ax=ax)
    ax.set_ylabel(None)  # Use ylabel from left plot
    ax.set_xlabel('Num Grad Steps')
    ax.set_title(r'$90^{\circ}$')
    plot_path = os.path.join(plot_dir,
                             f'max_grid_score_vs_num_grad_steps_by_place_cell_rf.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_max_grid_score_vs_place_cell_rf_by_activation(
        runs_performance_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 8),
                             sharey=True, sharex=True)

    ax = axes[0]
    sns.stripplot(y="score_60_by_neuron_max",
                  x='place_cell_rf',
                  hue='activation',
                  data=runs_performance_df,
                  ax=ax,
                  size=3)
    ax.set_ylabel(f'Max Grid Score')
    ax.set_xlabel('Gaussian ' + r'$\sigma$' + ' (m)')
    ax.set_title(r'$60^{\circ}$')

    ax = axes[1]
    sns.stripplot(y="score_90_by_neuron_max",
                  x='place_cell_rf',
                  hue='activation',
                  data=runs_performance_df,
                  ax=ax,
                  size=3)
    ax.set_ylabel(None)  # Use ylabel from left plot
    ax.set_xlabel('Gaussian ' + r'$\sigma$')
    ax.set_title(r'$90^{\circ}$')
    plot_path = os.path.join(plot_dir,
                             f'max_grid_score_vs_place_cell_rf_by_activation.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


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
            x='score_60_by_neuron_max',
            y='score_90_by_neuron_max',
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

    plot_path = os.path.join(plot_dir,
                             f'max_grid_score_90_vs_max_grid_score_60_by_activation_and_rnn_type.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_neural_predictivity_vs_activity_participation_ratio_by_architecture_and_activation(
        trained_neural_predictivity_and_ID_df: pd.DataFrame,
        plot_dir: str):
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
    plt.xlabel('Activity Participation Ratio')
    plt.ylabel('Neural Predictivity')

    plot_path = os.path.join(plot_dir,
                             f'neural_predictivity_vs_activity_participation_ratio_by_architecture_and_activation.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_neural_predictivity_vs_rate_maps_participation_ratio_by_architecture_and_activation(
        trained_neural_predictivity_and_ID_df: pd.DataFrame,
        plot_dir: str):
    g = sns.scatterplot(
        x='rate_maps_participation_ratio',
        y='Trained',
        hue='Architecture',
        style='Activation',
        data=trained_neural_predictivity_and_ID_df, )
    g.legend(
        bbox_to_anchor=(1, 0.5),  # 1 on the x axis, 0.5 on the y axis
        loc='center left',  # Legend goes center-left of anchor
    )
    plt.xlabel('Rate Maps Participation Ratio')
    plt.ylabel('Neural Predictivity')

    plot_path = os.path.join(plot_dir,
                             f'neural_predictivity_vs_rate_maps_participation_ratio_by_architecture_and_activation.pdf')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)

    plot_path = os.path.join(plot_dir,
                             f'neural_predictivity_vs_rate_maps_participation_ratio_by_architecture_and_activation.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_neural_predictivity_vs_rate_maps_participation_ratio_by_architecture_and_activation_custom(
        trained_neural_predictivity_and_ID_df: pd.DataFrame,
        plot_dir: str):
    avg_over_seed_df = trained_neural_predictivity_and_ID_df.groupby(['Architecture', 'Activation']).agg({
        'Trained': 'first',
        'rate_maps_participation_ratio': ['mean', 'sem']
    }).reset_index()

    plt.close()
    # This is such a pain in the ass. We'll do it manually.
    for row_idx, row in avg_over_seed_df.iterrows():
        plt.errorbar(
            x=row['rate_maps_participation_ratio']['mean'],
            y=row['Trained']['first'],
            xerr=row['rate_maps_participation_ratio']['sem'],
            linestyle='',  # Necessary to prevent lines from being connected
            color=architecture_color_map[row['Architecture'][0]],
            marker=activation_marker_map[row['Activation'][0]])

    plt.xlabel('Participation Ratio of Rate Maps')
    plt.ylabel('Neural Predictivity')

    # Add custom legend.
    # https://stackoverflow.com/a/45141109/4570472
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]

    labels = []
    handles = []
    for arch, arch_color in architecture_color_map.items():
        labels.append(arch)
        handles.append(f("s", arch_color))
    for activation, activation_marker in activation_marker_map.items():
        labels.append(activation)
        handles.append(f(activation_marker, 'k'))
    plt.legend()

    plt.legend(
        handles, labels, framealpha=1,
        bbox_to_anchor=(1, 0.5),  # 1 on the x axis, 0.5 on the y axis
        loc='center left',  # Legend goes center-left of anchor
    )

    plot_path = os.path.join(plot_dir,
                             f'neural_predictivity_vs_rate_maps_participation_ratio_by_architecture_and_activation_custom.pdf')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)

    plot_path = os.path.join(plot_dir,
                             f'neural_predictivity_vs_rate_maps_participation_ratio_by_architecture_and_activation_custom.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_neural_predictivity_vs_rate_maps_rank_by_architecture_and_activation(
        trained_neural_predictivity_and_ID_df: pd.DataFrame,
        plot_dir: str):
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

    plot_path = os.path.join(plot_dir,
                             f'neural_predictivity_vs_rate_maps_rank_by_architecture_and_activation.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_participation_ratio_by_num_grad_steps(
        runs_augmented_histories_df: pd.DataFrame,
        plot_dir: str, ):
    plt.close()
    sns.lineplot(y="participation_ratio",
                 x='num_grad_steps',
                 data=runs_augmented_histories_df)
    plt.ylabel(f'Participation Ratio')
    plt.xlabel('Num Grad Steps')

    plot_path = os.path.join(plot_dir,
                             f'participation_ratio_by_num_grad_steps.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


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
    plot_path = os.path.join(plot_dir,
                             f'participation_ratio_vs_architecture_and_activation.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_percent_grid_cells_vs_place_cell_rf_by_threshold(
        augmented_percent_neurons_score60_above_threshold_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    """
    Plot percent of units (aggregating across all runs) with a 60d grid score
    above a threshold (Y) against place cell RF (X). Each threshold is plotted
    as a trace in a different hue.
    """
    plt.close()
    g = sns.lineplot(
        data=augmented_percent_neurons_score60_above_threshold_by_run_id_df,
        x='place_cell_rf',
        y='Percent',
        hue='Grid Score Threshold',
    )
    g.legend(
        title='Grid Score Threshold',  # For some reason, the legend title disappears, so add it manually.
        bbox_to_anchor=(1, 0.5),  # 1 on the x axis, 0.5 on the y axis
        loc='center left',  # Legend goes center-left of anchor
    )
    plt.xlabel(r'$\sigma$ (m)')
    plt.ylabel('% Grid Cells')

    plot_path = os.path.join(plot_dir,
                             f'percent_grid_cells_vs_place_cell_rf_by_threshold.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_percent_grid_cells_vs_place_cell_rf_vs_place_cell_ss_by_threshold(
        augmented_percent_neurons_score60_above_threshold_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    for threshold, group_df in augmented_percent_neurons_score60_above_threshold_by_run_id_df.groupby(
            'Grid Score Threshold'):
        sns.heatmap(
            pd.pivot_table(group_df, index='surround_scale', columns='place_cell_rf', values='Percent'),
            cmap='Spectral_r',
            vmin=0.,
            vmax=100.,
            linewidths=.5)
        plt.xlabel(r'$\sigma$ (m)')
        plt.ylabel(r'$s$')
        plt.title(f'% Grid Cells (Threshold={threshold})')

        plot_path = os.path.join(plot_dir,
                                 f'percent_grid_cells_vs_place_cell_rf_vs_place_cell_ss_by_threshold={threshold}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_percent_have_grid_cells_given_low_pos_decoding_err_vs_human_readable_sweep(
        runs_configs_with_scores_max_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold_in_cm: float = 6.):
    thresholds = [0.3, 0.8, 1.2]

    for threshold in thresholds:
        plt.close()
        runs_configs_with_scores_max_df[f'pos_decoding_err_below_{threshold}'] = \
            runs_configs_with_scores_max_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm

        runs_configs_with_scores_max_df[f'has_grid_d60'] \
            = runs_configs_with_scores_max_df['score_60_by_neuron_max'] > threshold

        fig, ax = plt.subplots(figsize=(32, 8))

        # ax = axes[0]
        sns.barplot(y="has_grid_d60",
                    x='human_readable_sweep',
                    data=runs_configs_with_scores_max_df,
                    ax=ax,
                    )
        ax.set_ylim(0., 1.)
        # ax.set_title(r'$60^{\circ}$')
        plt.ylim(0., 1.)
        # plt.title(r'$60^{\circ}$')
        ax.set_ylabel(
            f'Frac Runs\nMax Grid Score > {threshold} | Pos Err < {low_pos_decoding_err_threshold_in_cm} cm')
        ax.set_xlabel('')

        # ax = axes[1]
        # sns.barplot(y="has_grid_d90",
        #             x='human_readable_sweep',
        #             data=runs_configs_with_scores_max_df,
        #             ax=ax)
        # ax.set_ylim(0., 1.)
        # ax.set_title(r'$90^{\circ}$')
        # ax.set_ylabel(
        #     f'Frac Runs : Max Grid Score > {grid_score_d90_threshold} | Pos Err < {low_pos_decoding_err_threshold_in_cm}')
        # ax.set_xlabel('')

        plot_path = os.path.join(plot_dir,
                                 f'percent_have_grid_cells_given_low_pos_decoding_err_vs_human_readable_sweep_threshold={threshold}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_percent_type_lattice_cells_given_low_pos_decoding_err_vs_activation(
        runs_performance_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold_in_cm: float = 6.,
        grid_score_d60_threshold: float = 1.2,
        grid_score_d90_threshold: float = 1.4):
    plt.close()
    runs_performance_low_pos_decod_err_df = runs_performance_df[
        runs_performance_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm].copy()

    runs_performance_low_pos_decod_err_df['has_grid_d60'] \
        = runs_performance_low_pos_decod_err_df['score_60_by_neuron_max'] > grid_score_d60_threshold

    runs_performance_low_pos_decod_err_df['has_grid_d90'] \
        = runs_performance_low_pos_decod_err_df['score_90_by_neuron_max'] > grid_score_d90_threshold

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

    plot_path = os.path.join(plot_dir,
                             f'percent_type_lattice_cells_given_low_pos_decoding_err_vs_activation.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_percent_runs_with_grid_cells_pie(runs_configs_with_scores_max_df: pd.DataFrame,
                                          plot_dir: str):
    thresholds = [0.3, 0.8, 1.2]

    for threshold in thresholds:
        plt.close()

        grid_score_above_threshold_col = f'score_60_by_neuron_max_{threshold}'
        runs_configs_with_scores_max_df[grid_score_above_threshold_col] = \
            runs_configs_with_scores_max_df['score_60_by_neuron_max'] >= threshold

        num_runs_per_category = runs_configs_with_scores_max_df.groupby(grid_score_above_threshold_col)[
            grid_score_above_threshold_col].count()

        fig, ax = plt.subplots(figsize=(4, 3))

        ax.pie(
            x=num_runs_per_category.values,
            labels=num_runs_per_category.index.values,
            colors=['tab:blue' if label == True else 'tab:orange'
                    for label in num_runs_per_category.index.values],
            # shadow=True,
            autopct='%.1f%%')
        plt.title(f'Runs With (Possible) Grid Cells\nThreshold={threshold}\nNum. of Networks={len(runs_configs_with_scores_max_df)}')

        plot_path = os.path.join(plot_dir, f'percent_runs_with_grid_cells_pie_threshold={threshold}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_percent_runs_with_grid_cells_vs_grid_score_threshold(
        runs_configs_with_scores_max_df: pd.DataFrame,
        plot_dir: str):
    thresholds = np.linspace(0.35, 1.15, 150)
    percent_runs_above_threshold = np.zeros_like(thresholds)
    for threshold_idx, threshold in enumerate(thresholds):
        percent_runs_above_threshold[threshold_idx] = 100 * (
                runs_configs_with_scores_max_df['score_60_by_neuron_max'] > threshold).mean()

    y_label = f'% Runs with (Possible) Grid Cells (N={len(runs_configs_with_scores_max_df)})'
    plot_df = pd.DataFrame.from_dict({
        'Grid Score Threshold': thresholds,
        y_label: percent_runs_above_threshold
    })

    plt.close()
    sns.lineplot(
        data=plot_df,
        x='Grid Score Threshold',
        y=y_label)
    plt.ylim(0, 100)

    plot_path = os.path.join(plot_dir, f'percent_runs_with_grid_cells_vs_grid_score_threshold.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_percent_runs_with_low_pos_decoding_err_pie(runs_configs_df: pd.DataFrame,
                                                    plot_dir: str,
                                                    low_pos_decoding_err_threshold_in_cm: float = 6.):
    plt.close()

    pos_decoding_err_below_threshold_col = f'pos_decoding_err_below_{low_pos_decoding_err_threshold_in_cm}'
    runs_configs_df[pos_decoding_err_below_threshold_col] = \
        runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm

    num_runs_per_category = runs_configs_df.groupby(pos_decoding_err_below_threshold_col)[
        pos_decoding_err_below_threshold_col].count()

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.pie(
        x=num_runs_per_category.values,
        labels=num_runs_per_category.index.values,
        colors=['tab:blue' if label == True else 'tab:orange'
                for label in num_runs_per_category.index.values],
        # shadow=True,
        autopct='%.1f%%')
    plt.title(
        f'Runs with Low Position Error\nThreshold={low_pos_decoding_err_threshold_in_cm} cm, N={len(runs_configs_df)}')

    plot_path = os.path.join(plot_dir, f'percent_runs_with_low_pos_decoding_err_pie.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_percent_low_decoding_err_vs_human_readable_sweep(
        runs_configs_df: pd.DataFrame,
        plot_dir: str,
        low_pos_decoding_err_threshold_in_cm: float = 6.):
    plt.close()
    runs_configs_df[f'pos_decoding_err_below_{low_pos_decoding_err_threshold_in_cm}'] = \
        runs_configs_df['pos_decoding_err'] < low_pos_decoding_err_threshold_in_cm

    fig, ax = plt.subplots(figsize=(24, 8))
    sns.barplot(x="human_readable_sweep",
                y=f'pos_decoding_err_below_{low_pos_decoding_err_threshold_in_cm}',
                data=runs_configs_df,
                ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(f'Frac Runs : Pos Error < {low_pos_decoding_err_threshold_in_cm} cm')
    ax.set_ylim(0., 1.)
    plt.savefig(os.path.join(plot_dir,
                             f'percent_low_decoding_err_vs_human_readable_sweep.png'),
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
                    x='score_60_by_neuron_max',
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
                    x='score_90_by_neuron_max',
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

    plot_path = os.path.join(plot_dir,
                             f'pos_decoding_error_vs_num_grad_steps_by_place_cell_rf.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_pos_decoding_err_min_vs_activation(runs_configs_df: pd.DataFrame,
                                            plot_dir: str):
    plt.close()
    sns.stripplot(y="pos_decoding_err",
                  x='activation',
                  data=runs_configs_df,
                  )
    plt.axhline(y=100., color='r', linewidth=5)
    plt.text(x=0, y=55, s='Untrained', color='r')
    plt.ylim(0.1, 100.)
    plt.yscale('log')
    plt.ylabel(f'Pos Decoding Err (cm)')
    plt.xlabel('')
    plot_path = os.path.join(plot_dir,
                             f'pos_decoding_err_min_vs_activation.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


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
    plot_path = os.path.join(plot_dir,
                             f'pos_decoding_err_min_vs_architecture.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


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
    plot_path = os.path.join(plot_dir,
                             f'pos_decoding_err_min_vs_optimizer.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


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
    ax.set_yscale("log")
    ax.set_ylabel('Pos Decoding Err (cm)')
    ax.set_xlabel('')
    plot_path = os.path.join(plot_dir,
                             f'pos_decoding_err_vs_human_readable_sweep.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_rate_maps_by_run_id(
        neurons_data_by_run_id_df: pd.DataFrame,
        joblib_files_data_by_run_id_dict: Dict[str, Dict[str, np.ndarray]],
        plot_dir: str,
        smooth: bool = True):
    for run_id, run_id_neurons_df in neurons_data_by_run_id_df.groupby(['run_id']):

        n_rows = n_cols = 5

        fig, axes = plt.subplots(
            n_rows,  # rows
            n_cols,  # columns
            figsize=(2 * n_rows, 2 * n_cols),
            sharey=True,
            sharex=True,
            gridspec_kw={'width_ratios': [1] * n_cols})

        run_id_neurons_df.sort_values(
            'score_60_by_neuron',
            ascending=False,
            inplace=True)

        for ax_idx, (row_idx, row) in enumerate(run_id_neurons_df.iterrows()):
            run_id = row['run_id']
            neuron_idx = row['neuron_idx']
            score_60 = row['score_60_by_neuron']
            rate_map = joblib_files_data_by_run_id_dict[run_id]['rate_maps'][neuron_idx]

            if smooth:
                rate_map = np.copy(rate_map)
                rate_map[np.isnan(rate_map)] = 0.
                rate_map = scipy.ndimage.gaussian_filter(rate_map, sigma=2.)

            row, col = ax_idx // n_cols, ax_idx % n_cols
            ax = axes[row, col]

            sns.heatmap(
                data=rate_map,
                # vmin=np.nanmin(rate_maps[storage_idx]),
                # vmax=np.nanmax(rate_maps[storage_idx]),
                ax=ax,
                cbar=False,
                cmap='Spectral_r',
                square=True,
                yticklabels=False,
                xticklabels=False)

            ax.set_title(f'{np.round(score_60, 2)}')

            # Seaborn's heatmap flips the y-axis by default. Flip it back ourselves.
            ax.invert_yaxis()

            if ax_idx == 24:
                break

        plt.tight_layout()
        plot_path = os.path.join(plot_dir,
                                 f'rate_maps_by_run_id={run_id}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_rate_maps_examples_hexagons_by_score_range(
        neurons_data_by_run_id_df: pd.DataFrame,
        joblib_files_data_by_run_id_dict: Dict[str, Dict[str, np.ndarray]],
        plot_dir: str,
        max_num_ratemaps_per_range: int = 12,
        smooth: bool = True):
    grid_score_ranges = [
        (0., 0.25),
        (0.25, 0.35),
        (0.35, 0.45),
        (0.45, 0.65),
        (0.65, 0.80),
        (0.80, 0.90),
        (0.90, 1.15),
        (1.15, 10),
    ]

    # n_rows = n_cols = int(np.ceil(np.sqrt(max_num_ratemaps_per_range)))
    n_cols = 4
    n_rows = int(np.ceil(max_num_ratemaps_per_range / n_cols))

    for grid_score_range in grid_score_ranges:

        low, high = grid_score_range

        indices = (neurons_data_by_run_id_df['score_60_by_neuron'] >= low) \
                  & (neurons_data_by_run_id_df['score_60_by_neuron'] < high)

        neurons_in_range_df = neurons_data_by_run_id_df[indices]
        neurons_to_plot_df = neurons_in_range_df.sample(
            n=min(n_cols * n_rows, len(neurons_in_range_df)),
            replace=False,
            random_state=0,  # for reproducibility
        )

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(2 * n_rows, 2 * n_cols),
            sharey=True,
            sharex=True,
            gridspec_kw={'width_ratios': [1] * n_cols})

        for ax_idx, (row_idx, row) in enumerate(neurons_to_plot_df.iterrows()):
            run_id = row['run_id']
            neuron_idx = row['neuron_idx']
            score_60 = row['score_60_by_neuron']
            rate_map = joblib_files_data_by_run_id_dict[run_id]['rate_maps'][neuron_idx]

            if smooth:
                rate_map = np.copy(rate_map)
                rate_map[np.isnan(rate_map)] = 0.
                rate_map = scipy.ndimage.gaussian_filter(rate_map, sigma=2.)

            row, col = ax_idx // n_cols, ax_idx % n_cols
            ax = axes[row, col]

            sns.heatmap(
                data=rate_map,
                # vmin=np.nanmin(rate_maps[storage_idx]),
                # vmax=np.nanmax(rate_maps[storage_idx]),
                ax=ax,
                cbar=False,
                cmap='Spectral_r',
                square=True,
                yticklabels=False,
                xticklabels=False)

            ax.set_title(f'{np.round(score_60, 2)}')

            # Seaborn's heatmap flips the y-axis by default. Flip it back ourselves.
            ax.invert_yaxis()

        # Replace any empty axes with empty heatmaps.
        empty_rate_map = np.full_like(rate_map, fill_value=np.nan)
        for ax_idx in range(ax_idx, n_rows * n_cols):
            row, col = ax_idx // n_cols, ax_idx % n_cols
            ax = axes[row, col]
            sns.heatmap(
                data=empty_rate_map,
                ax=ax,
                cbar=False,
                cmap='Spectral_r',
                square=True,
                yticklabels=False,
                xticklabels=False)

        plt.tight_layout()

        plot_path = os.path.join(plot_dir,
                                 f'rate_maps_examples_hexagons_by_score_range_low={low}_high={high}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_rate_maps_examples_hexagons_by_score_sorted(
        neurons_data_by_run_id_df: pd.DataFrame,
        joblib_files_data_by_run_id_dict: Dict[str, Dict[str, np.ndarray]],
        plot_dir: str,
        max_num_ratemaps_per_range: int = 12,
        smooth: bool = True):
    # n_rows = n_cols = int(np.ceil(np.sqrt(max_num_ratemaps_per_range)))
    n_cols = 4
    n_rows = int(np.ceil(max_num_ratemaps_per_range / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(2 * n_rows, 2 * n_cols),
        sharey=True,
        sharex=True,
        gridspec_kw={'width_ratios': [1] * n_cols})
    #
    # neurons_data = neurons_data_by_run_id_df[['run_id']]

    # for ax_idx, (row_idx, row) in enumerate(neurons_data_by_run_id_df.iterrows()):
    #     run_id = row['run_id']
    #     neuron_idx = row['neuron_idx']
    #     score_60 = row['score_60_by_neuron']
    #     rate_map = joblib_files_data_by_run_id_dict[run_id]['rate_maps'][neuron_idx]
    #
    #     if smooth:
    #         rate_map = np.copy(rate_map)
    #         rate_map[np.isnan(rate_map)] = 0.
    #         rate_map = scipy.ndimage.gaussian_filter(rate_map, sigma=2.)
    #
    #     row, col = ax_idx // n_cols, ax_idx % n_cols
    #     ax = axes[row, col]
    #
    #     sns.heatmap(
    #         data=rate_map,
    #         # vmin=np.nanmin(rate_maps[storage_idx]),
    #         # vmax=np.nanmax(rate_maps[storage_idx]),
    #         ax=ax,
    #         cbar=False,
    #         cmap='Spectral_r',
    #         square=True,
    #         yticklabels=False,
    #         xticklabels=False)
    #
    #     ax.set_title(f'{np.round(score_60, 2)}')
    #
    #     # Seaborn's heatmap flips the y-axis by default. Flip it back ourselves.
    #     ax.invert_yaxis()
    #
    # # Replace any empty subplots with empty heatmaps
    # empty_rate_map = np.full_like(rate_map, fill_value=np.nan)
    # for ax_idx in range(ax_idx, n_rows * n_cols):
    #     row, col = ax_idx // n_cols, ax_idx % n_cols
    #     ax = axes[row, col]
    #     sns.heatmap(
    #         data=empty_rate_map,
    #         ax=ax,
    #         cbar=False,
    #         cmap='Spectral_r',
    #         square=True,
    #         yticklabels=False,
    #         xticklabels=False)
    #
    # plt.tight_layout()
    #
    # plot_path = os.path.join(plot_dir,
    #                          f'rate_maps_examples_hexagons_by_score_sorted.png')
    # plt.savefig(plot_path,
    #             bbox_inches='tight',
    #             dpi=300)
    plt.show()
    plt.close()
    # print(f'Plotted {plot_path}')
    raise NotImplementedError


def plot_rate_maps_examples_squares_by_score_range(
        neurons_data_by_run_id_df: pd.DataFrame,
        joblib_files_data_by_run_id_dict: Dict[str, Dict[str, np.ndarray]],
        plot_dir: str,
        max_num_ratemaps_per_range: int = 12,
        smooth: bool = True):
    grid_score_ranges = [
        (1.1, 1.2),
        (1.2, 1.3),
        (1.3, 1.4),
        (1.4, 1.5),
        (1.5, 1.6),
        (1.6, 10),
    ]

    # n_rows = n_cols = int(np.ceil(np.sqrt(max_num_ratemaps_per_range)))
    n_cols = 4
    n_rows = int(np.ceil(max_num_ratemaps_per_range / n_cols))

    for grid_score_range in grid_score_ranges:

        low, high = grid_score_range

        indices = (neurons_data_by_run_id_df['score_90_by_neuron'] >= low) \
                  & (neurons_data_by_run_id_df['score_90_by_neuron'] < high)

        neurons_in_range_df = neurons_data_by_run_id_df[indices]
        neurons_to_plot_df = neurons_in_range_df.sample(
            n=min(n_cols * n_rows, len(neurons_in_range_df)),
            replace=False,
            random_state=0,  # for reproducibility
        )

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(2 * n_rows, 2 * n_cols),
            sharey=True,
            sharex=True,
            gridspec_kw={'width_ratios': [1] * n_cols})

        for ax_idx, (row_idx, row) in enumerate(neurons_to_plot_df.iterrows()):
            run_id = row['run_id']
            neuron_idx = row['neuron_idx']
            score_90 = row['score_90_by_neuron']
            rate_map = joblib_files_data_by_run_id_dict[run_id]['rate_maps'][neuron_idx]
            if smooth:
                rate_map = np.copy(rate_map)
                rate_map[np.isnan(rate_map)] = 0.
                rate_map = scipy.ndimage.gaussian_filter(rate_map, sigma=2.)
            row, col = ax_idx // n_cols, ax_idx % n_cols
            ax = axes[row, col]

            sns.heatmap(
                data=rate_map,
                # vmin=np.nanmin(rate_maps[storage_idx]),
                # vmax=np.nanmax(rate_maps[storage_idx]),
                ax=ax,
                cbar=False,
                cmap='Spectral_r',
                square=True,
                yticklabels=False,
                xticklabels=False)

            ax.set_title(f'{np.round(score_90, 2)}')

            # Seaborn's heatmap flips the y-axis by default. Flip it back ourselves.
            ax.invert_yaxis()

        # Replace any empty subplots with empty heatmaps
        empty_rate_map = np.full_like(rate_map, fill_value=np.nan)
        for ax_idx in range(ax_idx, n_rows * n_cols):
            row, col = ax_idx // n_cols, ax_idx % n_cols
            ax = axes[row, col]
            sns.heatmap(
                data=empty_rate_map,
                ax=ax,
                cbar=False,
                cmap='Spectral_r',
                square=True,
                yticklabels=False,
                xticklabels=False)

        plt.tight_layout()

        plot_path = os.path.join(plot_dir,
                                 f'rate_maps_examples_squares_by_score_range_low={low}_high={high}.png')
        plt.savefig(plot_path,
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
        print(f'Plotted {plot_path}')


def plot_square_scores_histograms_by_activation(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    # 97 chosen to match bin width of grid scores
    bins = np.linspace(-0.6, 2.0, 97)
    g = sns.histplot(
        data=augmented_neurons_data_by_run_id_df,
        x='score_90_by_neuron',
        bins=bins,
        kde=True,
        hue='activation')
    g.legend_.set_title('Activation')
    # sns.displot(
    #     data=augmented_neurons_data_by_run_id_df,
    #     x='score_60_by_neuron',
    #     bins=bins,
    #     kde=True,
    #     hue='n_place_fields_per_cell',
    #     col='n_place_fields_per_cell'
    # )

    plt.xlabel('Square Score')
    plt.ylabel('Number of Units')
    plot_path = os.path.join(plot_dir,
                             f'square_scores_histograms_by_activation.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')


def plot_square_scores_histograms_by_optimizer(
        augmented_neurons_data_by_run_id_df: pd.DataFrame,
        plot_dir: str):
    plt.close()
    # 97 chosen to match bin width of grid scores
    bins = np.linspace(-0.6, 2.0, 97)
    g = sns.histplot(
        data=augmented_neurons_data_by_run_id_df,
        x='score_90_by_neuron',
        bins=bins,
        kde=True,
        hue='optimizer')
    g.legend_.set_title('Optimizer')
    # sns.displot(
    #     data=augmented_neurons_data_by_run_id_df,
    #     x='score_60_by_neuron',
    #     bins=bins,
    #     kde=True,
    #     hue='n_place_fields_per_cell',
    #     col='n_place_fields_per_cell'
    # )

    plt.xlabel('Square Score')
    plt.ylabel('Number of Units')
    plot_path = os.path.join(plot_dir,
                             f'square_scores_histograms_by_optimizer.png')
    plt.savefig(plot_path,
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()
    print(f'Plotted {plot_path}')
