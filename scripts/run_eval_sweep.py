import os
import random
import subprocess
import time
import wandb


### Option 1: Manually specify sweep IDs, then fetch (finished) run IDs.

# Cartesian (Low Dim)
# sweep_ids = ['gvxvhnx8']

# Cartesian (High Dim)
sweep_ids = ['2ks3s65c']

# Gaussian
# sweep_ids = ['oa0v2uzr']

# DoG (true DoG)
# sweep_ids = ['nisioabg']

# DoS (done)
# sweep_ids = ['vxbwdefk']

# Nayebi sweep (done)
# sweep_ids = ['59lptrr1']


# Fetch runs associated with the relevant sweeps.
api = wandb.Api(timeout=60)
run_ids = []
for sweep_id in sweep_ids:
    sweep = api.sweep(f'rylan/mec-hpc-investigations/{sweep_id}')
    sweep_run_ids = [run.id for run in sweep.runs if run.state == 'finished']
    run_ids.extend(sweep_run_ids)
print(f'Collected runs from sweeps: {sweep_ids}')


### Option 2: Manually specify run IDs.

# All runs.
# results_dir = 'results'
# run_ids = list(sorted(os.listdir(results_dir)))

# Specific runs
# run_ids = []


# random.shuffle functions in-place.
random.shuffle(run_ids)

print(f'Run IDs: {run_ids}')

for idx, run_id in enumerate(run_ids):

    command_and_args = [
        'sbatch',
        'scripts/run_eval_one.sh',
        run_id]

    print(' '.join(command_and_args))
    subprocess.run(command_and_args)

    # if idx > 100:
    #     break
