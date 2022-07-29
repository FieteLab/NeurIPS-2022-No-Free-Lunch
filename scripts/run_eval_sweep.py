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
run_ids = ['7a7ktmwd', 'k97fuh2w', '7yiodipq', '1eohgoy4', 'k3byw15i', 'aldnn6ql', '59jrwni9', '0f0pk2yu', '1hi5xlxk', 'pp8l1cfs', 'yzu7wvhq', '5yp08mxp', '54e6d3x4', 'uneylmc8', 'jhm2hlga', '08fvp5z2', 'ol1sdy3c', 'l9gu1gkl', 'x71lcqmn', 'q6trd8sn', 'khnuf5vf', 'zmdrezdg', '9h7r2te0', 'r7ukfb8q', 'u9daz8op', 'v3x7k5bh', '9moeo4d9', '7xl0a33i', 'kq0t3ziz', 'jx6ct9k8', '7nyrs5gz', 'q7qsoqxh', 'um6ybsqf', 'jjicdfjt', 'nyds2zx3', '4m7b8341', '0prrnf5c', 'vvsv8yio', '0ues74ev', 'xiwf84un', 'vi2bjylp', 'rlc7ussj', 'vjffi0hh', 'uona1czh', 'sp9s61ka', 'ri04melu', 'gaaf5il5', 'iumf02v5', 'plyvqx1z', 'xva3aio5', 'fmr3tr0l', 'ia47vots', '37stamsv', 'lqp3x7d6', 'n28o0pzx', 'lzg3064v', 'lnhgn7n0', '5s2tusnp', 'awwks2ey', '1pvp86fj', 'i4sqiwi0', 'weein0e7', 'mfo0eiop', '19z6uit4', 'xgsgm3yu', '1yhie0ho', 'pgv6mm3g', '0g62e5m2', '6hysewph', 'nyt27nj5', 'irh2l6n9', 'ykdgxamn', 'wptw3cwd', 'n3hqgelv', 'yqu3ravz', 'ahlfhiz1', 'c6qhkk0s', 'ake60d2a', 'i6tx0ksu', 'pko5bxkd', 'rekcbh8i', '67hyxs25', 'cb9hhbti', '7fv2jdsh', 'mtdfd1wr', 'sghgrvbn', '238do866', 'cyy6vnfn', 'a06uadxs', '4en4p2k7', 'wvgl1z36', '053p9hio', '4b96qgrl', 'i7tmwus6', '5a8h2evr', 'xd4h8iyn', 'yjbwe36i', 'bxfzf6cn', '4su86qo1', 'j84hq6t0', 'aprzp6wb', '6xxr9dt6', 'tq2x1wse', 'e5tej245', 'lhgwylps', '5dat50hu', 'cxt51q65', 'sp422uz2', '475jhisv', 'v283jtbj', 'x2vw3fo6', 'blhyp83h', 'pfc7o056', 'xwn595jy', 'trf3zk8x', 'j57angtv', 'khvfxr8v', '6pyuz2yc', '81z7rier', 'z7p84v3c', 'thbub652', 'is4ex92q', 'bkdackkh', 'iyeugj20']


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
