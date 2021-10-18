import time
from pathlib import Path
from nni.experiment import Experiment

# Search Space
search_space = {
    "tcl_num":
        {"_type": "choice", "_value": [1, 2, 3]},
    "tcl_channel_size":
        {"_type": "choice", "_value": [8, 16, 24, 32]},
    "kernel_size":
        {"_type": "choice", "_value": [3, 5, 7]},
    "dropout":
        {"_type": "choice", "_value": [0, .1, .2, .4]},
    "slices":
        {"_type": "choice", "_value": [1, 2]},
    "use_bias":
        {"_type": "choice", "_value": [True, False]},
    "lr":
        {"_type": "choice", "_value": [.01, .005, .0001]}
}

max_trials = 3_00

# Search Configuration
search = Experiment('local')
# Search Name
search.config.experiment_name = 'Rain Classifier'
search.config.trial_concurrency = 2
search.config.max_trial_number = max_trials
search.config.search_space = search_space
search.config.trial_command = 'python3 trial.py'
search.config.trial_code_directory = Path(__file__).parent
# Search Tuner Settings
search.config.tuner.name = 'Evolution'
search.config.tuner.class_args['optimize_mode'] = 'minimize'
search.config.tuner.class_args['population_size'] = 32

# Running Search
search.start(8080)

# Awaiting Results
executed_trials = 0
while True:
    trials = search.export_data()
    if executed_trials != len(trials):
        executed_trials = len(trials)
        print(f'\nTrials: {executed_trials} / {max_trials}', end = "")
    if search.get_status() == 'DONE':
        best_trial = min(trials, key = lambda t: t.value)
        print(f'\nBest trial params: {best_trial.parameter}')
        input("Experiment is finished. Press any key to exit...")
        break
    print('.', end = ""),
    time.sleep(10)
