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


# # Fetch runs associated with the relevant sweeps.
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
run_ids = ['xr8so4ls', 'fmyw73by', 'p0e1tg0g', 'ekmq5ijm', '5i5g42qx', 'jmafm0x5', 'pxu0hid3', 'a4he41iw', 'ugtsy1hv', '2xcycf9v', '1amcym9l', 'cyxe0wwh', '59k504w4', 'y3csn9cf', '61ii98hp', 'dn3je3c1', 'l51ilia3', '6k4208cu', 'y34iq5on', 'h1g92bwa', 'ykkx8vhs', 'i0jgx9gj', 'spusyr5v', 'zu19ynk8', 'sypcyou2', 'kq40nfyz', 'okowtmwe', 'tk5nea8u', '015tfkuz', 'cbiqva26', 'v0rgu757', 'y48zzrk8', 'yeuxs9h5', 'ank336jh', 'd2k9of0n', 'quqlt6rd', 'kj92sehk', 'ymx1zavm', 'b7121ifx', '8ao4t1y8', 'pn1kkl8y', 'zye01mi7', '6jidf78n', 'bv7a23wn', 'tryee22h', 'mjfufbgw', 'rcoajiiu', 'c3s8krv8', '8iwo30mx', 'tqimuvqp', 'mmgforyo', 'nkz6r699', 'i58vfs2h', '1xbnrltu', 'nj1ewm6w', '8biuafs2', 'et8t3zqk', '0qyc5xx1', 'q9l70z1y', 'r5pcv3wp', 'b9tlor1c', 'ietefxzn', 'ted8jx7t', 'efy0qx8f', 'gw1xhmnf', 'ejtschyv', 'tspkhihn', 'pz6stxcn', '8yrrymrr', 'vtyx1gk2', 'ek9aneyy', '18qqpar9', 'mqg3fhln', 'wp3si45y', '51fw15ig', '8v35mn6w', 'wpxqeodb', 'een6fde0', 'l2ug04st', 'ydgbmay1', 'gpwpz6ai', '2e7ym98t', 'd9rspmic', 'ows8foup', '0sgm7iz0', 'cg8q0162', '1jdho6zh', 'weaj81h6', 'dhq1lznu', 'tfti4os9', 'kaj1y2ja', 'm27lagpb', '0ev9jj9m', '7sinxki2', 'vljq5vz4', 'xof87uon']


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
