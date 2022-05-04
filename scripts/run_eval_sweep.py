import os
import subprocess


results_dir = 'results'
run_ids = list(os.listdir(results_dir))

for run_id in run_ids:
    command_and_args = [
        'sbatch',
        'scripts/run_eval_one.sh',
        run_id]

    print(f'Launching ' + ' '.join(command_and_args))
    subprocess.run(command_and_args)
    print(f'Launched ' + ' '.join(command_and_args))
    break