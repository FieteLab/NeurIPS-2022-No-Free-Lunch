import os
import subprocess


results_dir = 'results'
run_ids = list(sorted(os.listdir(results_dir)))

# Different optimizers (good position decoding only).
# run_ids = ['cve7h1a3', 'eb7okxma', 'gg4s24fd', 'd3i31lo5', 'id3fq1io', 'e9ss880d', '1ejiqtwp', 'c1pe8vib']

# Different architectures (good position decoding only)
# run_ids = ['ysaxu030', 'r5hysyoa', 'l3jipv4s', 'strxn0n2']

for idx, run_id in enumerate(run_ids):
    command_and_args = [
        'sbatch',
        'scripts/run_eval_one.sh',
        run_id]

    subprocess.run(command_and_args)
    print(f'Launched ' + ' '.join(command_and_args))

    if idx > 30:
        break
