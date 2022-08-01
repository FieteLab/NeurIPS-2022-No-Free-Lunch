import os
import random
import subprocess
import time
import wandb


### Option 1: Manually specify sweep IDs, then fetch (finished) run IDs.

# Cartesian (Low Dim)
# sweep_ids = ['gvxvhnx8']

# Cartesian (High Dim)
# sweep_ids = ['2ks3s65c']

# Polar
sweep_ids = ['m10yfzgz']

# Gaussian
# sweep_ids = ['oa0v2uzr']

# DoG (true DoG)
# sweep_ids = ['nisioabg']

# DoS (done)
# sweep_ids = ['vxbwdefk']

# DoS Multiple Scale
# sweep_ids = ['rwb622oq']

# DoS Multiple field & Multiple scale
# sweep_ids = ['lk012xp8', '2lj5ngjz']

# Nayebi sweep (done)
# sweep_ids = ['59lptrr1']

# Ideal DoG
# sweep_ids = ['bav6z2py']


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
run_ids = ['0ro6eo9s', 'iny3vrhc', 'nau5pr5d', 'lsq216i7', 'qblmlwlc', '1l49bsit', 'v0lhwrf0', '0cm2nfwe', '6orcbkzw', '4t5wkwnc', 'zlxa5fkb', 'snllejtr', '7jgdgewo', 'gek19gjq', 'b53i6hy9', '76x03hu2', 'rmtx3kf0', 'ks7s6adp', 'nw9pit9l', 'tre2mgvw', 'le50ezwc', 'v81h0dyq', 'jgpyvm83', 'dmyd87fh', 'cl16efzi', 'uhbqyf5s', '8f8u5dyx', 'noy7wye5', 'n6ob6ulz', '99vu6tu4', 'wz3v82g5', 'i8jmfygt', 'jwzhc1ad', 'pslomn1y', '0f819ylb', 'y1yvhp6t', 'wvic6qza']


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
