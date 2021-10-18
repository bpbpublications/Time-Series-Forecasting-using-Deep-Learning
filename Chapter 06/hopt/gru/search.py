import time
from pathlib import Path
from nni.experiment import Experiment

# Search Space
search_space = {
    "optimizer":
        {"_type": "choice", "_value": ['adam', 'sgd', 'adamax']},
    "gru_hidden_size":
        {"_type": "choice", "_value": [8, 12, 16, 24, 32]},
    "learning_rate":
        {"_type": "choice", "_value": [.001, .005, .01]}
}

max_trials = 30

# Search Configuration
search = Experiment('local')
# Search Name
search.config.experiment_name = 'GRU Search'
search.config.trial_concurrency = 2
search.config.max_trial_number = max_trials
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
