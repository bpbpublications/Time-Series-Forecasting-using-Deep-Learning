import time
from pathlib import Path
from nni.experiment import Experiment

# Search Space
search_space = {
    "trend_filter":       {
        "_type":  "choice",
        "_value": [{"_name": "None"}, {"_name": "hp"}, {"_name": "cf"}]
    },
    "cycle_filter":       {
        "_type":  "choice",
        "_value": [{"_name": "None"}, {"_name": "hp"}, {"_name": "cf"}]
    },
    "casual_convolution": {
        "_type":  "choice",
        "_value": [
            {
                "_name": False
            },
            {
                "_name":  True,
                "kernel": {"_type": "choice", "_value": [3, 5, 7, 9]},
            },
        ]
    },
    "rnn_hidden_size":    {"_type": "choice", "_value": [8, 16, 24]},
    "fcnn_layer_num":     {"_type": "choice", "_value": [0, 1, 2]},
    "fcnn_layer_size":    {"_type": "choice", "_value": [4, 8, 12]},
}

max_trials = 300

# Search Configuration
search = Experiment('local')
# Search Name
search.config.experiment_name = 'Hybrid Search'
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
        print(f'\nTrials: {executed_trials} / {max_trials}')
    if search.get_status() == 'DONE':
        best_trial = min(trials, key = lambda t: t.value)
        print(f'Best trial params: {best_trial.parameter}')
        input("Experiment is finished. Press any key to exit...")
        break
    print('.', end = ""),
    time.sleep(10)
