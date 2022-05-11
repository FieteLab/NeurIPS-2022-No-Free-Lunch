import os
import subprocess
import time
import wandb


### Option 1: Manually specify sweep IDs, then fetch (finished) run IDs.

# DoG, sweeping RF.
# sweep_ids = ['yzszqr74']

# DoG, multiple scales.
# sweep_ids = ['2yfpvx86']

# DoG, sweeping architectures.
# sweep_ids = ['can8n6vd']


# # Fetch runs associated with the relevant sweeps.
# api = wandb.Api(timeout=60)
# run_ids = []
# for sweep_id in sweep_ids:
#     runs = api.runs(path='mec-hpc-investigations', filters={"Sweep": sweep_id})
#     sweep_run_ids = [run.id for run in runs.objects if run.state == 'finished']
#     print(sweep_run_ids)
#     # TODO: why do I need to run this manually myself?
#     run_ids.extend(sweep_run_ids)


### Option 2: Manually specify run IDs.

# All runs.
results_dir = 'results'
run_ids = list(sorted(os.listdir(results_dir)))

# Ideal grid cells
# run_ids = ['wxt06g20', 'y5qdmmqx', 'ryrmls1x', 'otuv2dhn', 'kfpr44o9', '08jmt76g', '1ez9xulc', '6lgoiwhw', '0svwod2a', 'zg5hbvxx', 'ebb8dp9b', 'd47g0wpn', 'goo0np7q', 'qg3a3h8e', 'p0osju5b', 'ltzh0j9x']

# Different optimizers (good position decoding only).
# run_ids = ['cve7h1a3', 'eb7okxma', 'gg4s24fd', 'd3i31lo5', 'id3fq1io', 'e9ss880d', '1ejiqtwp', 'c1pe8vib']

# DoG with various receptive fields
# run_ids = ['95esrihf', 'f6tu5v07', 'c7fihyol', 'utmf94mn', 'iyv6e74j', 'tggeb3yy', 'hs2n1a57', '5n9wle3f', '8oisxj1q', 'lrtcifh3', '6vc4am5a', '29w2bqje', 'c4vn8r1j', 'tq91ttdd', 'ft6fenkc', '53zwi7kf', 'pym3d3du', 'tjzkib13', '6boglxlg', 'bilspnmo', 'mugg8n8f', 'nu2em0g9', '5xs8uja4', 'qfa5ur2h', '1m1q7e2y', 'ed9rccmx', 'z9obb0ry', 'aivijea1', 'tbag7mx0', '99kwrc1j', 'hgjzbnw0', '20mjeqn3', 'yo4r34ao', 'n7syshxb', 'tcqow4co', '2uttb04y']

# run_ids = ['8nfxn42y', 'yo4r34ao', 'n7syshxb', 'tcqow4co', '2uttb04y']

# Different architectures.
# run_ids = ['fuocwcs5', '6a0b77a4', '64ymbvmu', 'i3bckef0', 'kofij0zt', 'xoc1k5xp', '03n18c8v', 'ysaxu030' 'r5hysyoa', 'l3jipv4s', 'strxn0n2',
#            'a8w0rsdv', '2tdtiomu', 'f7b216ba', 'xu7bzubn']

# run_ids = ['gz4eckug', 'q2z2kjdj', 'k8j64q4d', 'ayemnx3z', 'h5wmhr9j', 'nvvn5si1', 'l8mvqdau', 'gaqt4cge', 'vqkv6ydf', 'gsre9lrm', 's719h5rq', 'ruknw501', 'lq5llm9f', 'q1dl0ec0', '97hw6cwk', 'ozd5ysjg', 'rc4v6o59', 'ux4itzk8', 'zqkpmt91', 'froe4a97', 'tssk4wan', 'f60skvry', 'pxn35xav', 'zqij896k', 'e22br48u', 'mjlpd2o9', 'vam2ebhd', 'tmeyino3', 'fqqbs71l', 'tbb38yx3', 'oqbqnszk', 'tqm7u4ar', 'uh7vr6ug', 'urtelgc3', 'b0r25j5n', 'vhazogal', 'oy9uxfxx', 'ae2xaehu', 'jqbr1tqd', 'd9ixjpr6', 'ghs072rb', 'v7ngiue5', 'medqm47k', 'sh52ccdk', 'n36d739l', 'iyj5x6e5', 'imrf7bwn', 'hd2n28hc', 'b35bwo1f', 'ik09eic9', '9asv7d9l', 'bgxcphb3', 'dhg7l53j', 'hg54oikt', '9xo7obn0', 'cdfl21hs', 'as06rgdl', 'y2gntktg', 'zzjnshz5', 'rn7nqgkf', 'hicl297p', 'm7dyxknv', 'ghjh651j', 'jrn4aiav', 'mcem8zjn', 'kvr4dcj5', 'o1kfwxz7', 'ciemxd0j', 'fwxoka8x', 'pzpv531u', 'hvxddwzk', 'kycldjto', 'rqvi2jxs', 'wwkojrwi', 'sv9cy0f0', 't8scqbwg', 'd80hfqel', 'qrrf14tv', 'gt9cts2z', 'h6tvolvw', '9r27d7g6', 'q1a06ld7', 'q5c4dkke', 'qtw5fq9m', 'c7822ae7', 'hawi5a6k', 'ielcw2q2', 'vp57wiur', 'jpiwyu6p', 'otzy5gnc', 'jzivpkgi', '6e82bcsw', '2mga3z9p', 'xntc8j0o', 'qmrqgb4v', 'zo4hnk3y', 'gdh8kdco', '8saxxd6d', 'lel8gbh4', 'kb434fcq']

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
