import os
import random
import subprocess
import time
import wandb


### Option 1: Manually specify sweep IDs, then fetch (finished) run IDs.

# Cartesian
# sweep_ids = ['gvxvhnx8']

# Gaussian
# sweep_ids = ['oa0v2uzr']

# DoG (true DoG)
sweep_ids = ['nisioabg']

# DoS (done)
# sweep_ids = ['vxbwdefk']

# Nayebi sweep (done)
# sweep_ids = ['59lptrr1']


# # Fetch runs associated with the relevant sweeps.
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
# run_ids = ['r8rp7sk6', 'fb01o9ti', 'oqe31nhf', '9tocnf46', 'fwxo3pzc', 'yc2i9ckn', '2spsic4d', 'fm75qhz5', '847ab7vi', 'ggepnwqx', 'g7b65vyr', 'tcbwci3a', 'r0dt5hin', '5blne46r', '4o2xywaf', '4ov9p5xf', '4z1tc7k4', '6d3shwgh', '9snwim33', 'asuo72ga', 'isttin3f', '8t2ukyhc', '3bvuz07h', '1nbk725k', '8hq05iug', '6rtdpbqc', 'ha2caztj', 'wa8tll44', '158jinf3', 'zk8jukpz', '4s5jlzoh', 'kq3860e3', '62ri1o4x', '4xqzth4j', 'zdljl6we', 'ck3t9e83', '9x65fo0x', 'e57mozth', 's2w63xp9', '85xpvtrx', 'rnqhf9ge', 'werrxbsm']


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
