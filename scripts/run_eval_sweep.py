import os
import subprocess
import wandb


### Option 1: Manually specify sweep IDs, then fetch (finished) run IDs.

# Different architectures.
sweep_ids = ['can8n6vd']


# Fetch runs associated with the relevant sweeps.
api = wandb.Api(timeout=60)
run_ids = []
for sweep_id in sweep_ids:
    runs = api.runs(path='mec-hpc-investigations',
                         filters={"Sweep": sweep_id})
    run_ids.extend([run.id for run in runs.objects if run.state == 'Finished'])

# print(run_ids)


### Option 2: Manually specify run IDs.

# All runs.
# results_dir = 'results'
# run_ids = list(sorted(os.listdir(results_dir)))

# Ideal grid cells
# run_ids = ['wxt06g20', 'y5qdmmqx', 'ryrmls1x', 'otuv2dhn', 'kfpr44o9', '08jmt76g', '1ez9xulc', '6lgoiwhw', '0svwod2a', 'zg5hbvxx', 'ebb8dp9b', 'd47g0wpn', 'goo0np7q', 'qg3a3h8e', 'p0osju5b', 'ltzh0j9x']

# Different optimizers (good position decoding only).
# run_ids = ['cve7h1a3', 'eb7okxma', 'gg4s24fd', 'd3i31lo5', 'id3fq1io', 'e9ss880d', '1ejiqtwp', 'c1pe8vib']

# DoG with various receptive fields
# run_ids = ['95esrihf', 'f6tu5v07', 'c7fihyol', 'utmf94mn', 'iyv6e74j', 'tggeb3yy', 'hs2n1a57', '5n9wle3f', '8oisxj1q', 'lrtcifh3', '6vc4am5a', '29w2bqje', 'c4vn8r1j', 'tq91ttdd', 'ft6fenkc', '53zwi7kf', 'pym3d3du', 'tjzkib13', '6boglxlg', 'bilspnmo', 'mugg8n8f', 'nu2em0g9', '5xs8uja4', 'qfa5ur2h', '1m1q7e2y', 'ed9rccmx', 'z9obb0ry', 'aivijea1', 'tbag7mx0', '99kwrc1j', 'hgjzbnw0', '20mjeqn3', 'yo4r34ao', 'n7syshxb', 'tcqow4co', '2uttb04y']

# run_ids = ['8nfxn42y', 'yo4r34ao', 'n7syshxb', 'tcqow4co', '2uttb04y']

# Different architectures.
run_ids = ['fuocwcs5', '6a0b77a4', '64ymbvmu', 'i3bckef0', 'kofij0zt', 'xoc1k5xp', '03n18c8v', 'ysaxu030' 'r5hysyoa', 'l3jipv4s', 'strxn0n2',
           'a8w0rsdv', '2tdtiomu', 'f7b216ba', 'xu7bzubn']

for idx, run_id in enumerate(run_ids):
    command_and_args = [
        'sbatch',
        'scripts/run_eval_one.sh',
        run_id]

    subprocess.run(command_and_args)
    print(f'Launched ' + ' '.join(command_and_args))

    # if idx > 100:
    #     break
