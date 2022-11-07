import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import shutil

notebook_dir = 'notebooks/40_pcpc_correlation'
results_dir = os.path.join(notebook_dir, 'results')
if os.path.exists(results_dir) and os.path.isdir(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir, exist_ok=True)


# According to Adrian:
# - Data was split in half to be sent over Slack.
# - Each array has shape (160,), then when you access each index, the
#   accessed array has shape (num PCs, num PCs).
# - The 320 total data are different mice in different environments on
#   different laps.

def roll_covariance(C: np.ndarray,
                    res: int = 50):

    Csquare = C.reshape(res, res, res, res)

    Cmean = np.zeros([res, res])
    for i in range(res):
        for j in range(res):
            Cmean += np.roll(np.roll(Csquare[i, j], -i, axis=0), -j, axis=1)

    Cmean = np.roll(np.roll(Cmean, res // 2, axis=0), res // 2, axis=1)

    return Cmean


for corrs_array_idx in range(1, 3):
    corrs_list = np.load(f'data/PCPC_Corrs{corrs_array_idx}.npy', allow_pickle=True)
    for i in range(5):
        plt.close()

        corrs = corrs_list[i]
        # identify all nan rows
        all_nan_rows = np.all(np.isnan(corrs), axis=1)
        not_nan_rows = ~all_nan_rows

        non_nan_corrs = corrs[not_nan_rows, :][:, not_nan_rows]

        Cmean = roll_covariance(C=non_nan_corrs)

        # Fourier transform
        Ctilde = np.fft.fft2(Cmean)

        # DC component of fourier transform
        # print(Ctilde[0,0])
        Ctilde[0, 0] = 0

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        # Plot
        ax = axes[0]
        ax.imshow(non_nan_corrs, cmap='jet', interpolation='gaussian')
        ax.set_xlabel(r'$\Sigma$', fontsize=20)

        ax = axes[1]
        ax.imshow(Cmean, cmap='jet', interpolation='gaussian')
        ax.set_xlabel(r'$\Sigma$ Rolled', fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1]
        width = 6
        idxs = np.arange(-width + 1, width)
        # idx = np.arange(width*2)
        x2, y2 = np.meshgrid(np.arange(2 * width - 1), np.arange(2 * width - 1))
        ax.scatter(x2, y2, c=np.abs(Ctilde)[idxs][:, idxs],
                   s=600, cmap='Oranges', marker='s',
                   #  norm=LogNorm(),
                   )
        ax.axis('square')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r'$\tilde\Sigma$', fontsize=20)
        axes[1].imshow(Ctilde)

        fig.suptitle(f'PC-PC Correlation (File: {corrs_array_idx}, Index: {i})')
        plt.savefig(os.path.join(results_dir,
                                 f'corrs{corrs_array_idx}_index={i}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
