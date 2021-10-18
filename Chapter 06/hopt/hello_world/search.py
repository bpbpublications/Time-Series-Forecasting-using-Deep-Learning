import time
from pathlib import Path
from nni.experiment import Experiment

# Search Space
search_space = {
    "x": {"_type": "choice", "_value": [-10, -5, 0, 5, 10]},
    "y": {"_type": "choice", "_value": [-10, -5, 0, 5, 10]},
    "z": {"_type": "choice", "_value": [-10, -5, 0, 5, 10]}
}

# Search Configuration
search = Experiment('local')
# Search Name
search.config.experiment_name = 'Hello World Search'
search.config.trial_concurrency = 4
search.config.max_trial_number = 50
search.config.search_space = search_space
search.config.trial_command = 'python3 trial.py'
search.config.trial_code_directory = Path(__file__).parent
# Search Tuner Settings
search.config.tuner.name = 'Evolution'
search.config.tuner.class_args['optimize_mode'] = 'minimize'
search.config.tuner.class_args['population_size'] = 8

# Running Search
search.start(8080)

# Awaiting Results
while True:
    if search.get_status() == 'DONE':
        trials = search.export_data()
        best_trial = min(trials, key = lambda t: t.value)
        print(f'Best trial params: {best_trial.parameter}')
        input("Experiment is finished. Press any key to exit...")
        break
    time.sleep(10)
