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


for corrs_array_idx in range(1, 3):
    corrs_list = np.load(f'data/PCPC_Corrs{corrs_array_idx}.npy', allow_pickle=True)
    for i in range(5):
        plt.close()
        corrs = corrs_list[i]
        # identify all nan rows
        all_nan_rows = np.all(np.isnan(corrs), axis=1)
        not_nan_rows = ~all_nan_rows

        plt.imshow(corrs[not_nan_rows, :][:, not_nan_rows])
        plt.title(f'PC-PC Correlation (File: {corrs_array_idx}, Index: {i})')
        plt.savefig(os.path.join(results_dir,
                                 f'corrs{corrs_array_idx}_index={i}.png'),
                    bbox_inches='tight',
                    dpi=300)
        # plt.show()
        plt.close()
