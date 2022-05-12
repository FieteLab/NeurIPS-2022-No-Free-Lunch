import os
import shutil
import subprocess
import time
import wandb


# All runs.
results_dir = 'results'
run_ids = list(sorted(os.listdir(results_dir)))

print(f'Run IDs: {run_ids}')

for run_id in run_ids:

    run_dir = os.path.join(results_dir, run_id)
    run_ckpts_dir = os.path.join(run_dir, 'ckpts')
    run_ckpts_dir_contents = os.listdir(run_ckpts_dir)
    if len(run_ckpts_dir_contents) == 8:
        continue
    assert 'ckpt-2.data-00000-of-00001' not in run_ckpts_dir_contents
    print(f'Deleting run_dir: {run_dir}')
    shutil.rmtree(run_dir)
