import wandb


api = wandb.Api()

run_ids = [
    # 'qs6l9zbf',
    'a8w0rsdv',
    '2tdtiomu',
    'f7b216ba',
    'xu7bzubn',
    'qco9lued',
    'strxn0n2',
    '6a0b77a4',
    '64ymbvmu',
    'i3bckef0',
    'kofij0zt',
]

new_sweep_id = 'can8n6vd'
sweep = api.sweep(f'rylan/mec-hpc-investigations/{new_sweep_id}')
for run_id in run_ids:
    run = api.run(f'rylan/mec-hpc-investigations/{run_id}')
    run.sweep = sweep
    run.update()
