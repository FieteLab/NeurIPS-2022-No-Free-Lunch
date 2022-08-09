import os
import random
import subprocess
import time
import wandb


### Option 1: Manually specify sweep IDs, then fetch (finished) run IDs.

# Cartesian (Low Dim)
# sweep_ids = ['gvxvhnx8']

# Cartesian (Low Dim) Random
# sweep_ids = ['koueijox']

# Cartesian (High Dim)
# sweep_ids = ['2ks3s65c']

# Cartesian (High Dim) Random
sweep_ids = ['iyycfgiv']

# Polar
# sweep_ids = ['m10yfzgz']

# Polar (High Dim) Random
# sweep_ids = ['2gz0ks4o']

# Gaussian (eval done)
# sweep_ids = ['oa0v2uzr']

# Gaussian Random
# sweep_ids = ['c2hohme2']

# DoG (true DoG)
# sweep_ids = ['nisioabg']

# DoS (done)
# sweep_ids = ['vxbwdefk']

# DoS Multiple Scale
# sweep_ids = ['rwb622oq']

# DoS Multiple field & Multiple scale
# sweep_ids = ['lk012xp8', '2lj5ngjz']

# Nayebi sweep (eval done)
# sweep_ids = ['59lptrr1']

# Ideal DoG (eval done)
# sweep_ids = ['bav6z2py']


# Fetch runs associated with the relevant sweeps.
# api = wandb.Api(timeout=60)
# run_ids = []
# for sweep_id in sweep_ids:
#     sweep = api.sweep(f'rylan/mec-hpc-investigations/{sweep_id}')
#     sweep_run_ids = [run.id for run in sweep.runs if run.state == 'finished']
#     run_ids.extend(sweep_run_ids)
# print(f'Collected runs from sweeps: {sweep_ids}')


### Option 2: Manually specify run IDs.

# All runs.
# results_dir = 'results'
# run_ids = list(sorted(os.listdir(results_dir)))

# Specific runs
run_ids = ['lymxsmpc']


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
