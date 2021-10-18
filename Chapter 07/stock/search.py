import time
from pathlib import Path
from nni.experiment import Experiment

# Search Space
fast_choices = {"_type": "choice", "_value": [3, 5, 7, 9]}
slow_choices = {"_type": "choice", "_value": [14, 20, 40]}
length_choices = {"_type": "choice", "_value": [5, 10, 20]}
ind_choices = [
    {"_name": "ao", "fast": fast_choices, "slow": slow_choices},
    {"_name": "apo", "fast": fast_choices, "slow": slow_choices},
    {"_name": "cci", "length": length_choices},
    {"_name": "cmo", "length": length_choices},
    {"_name": "mom", "length": length_choices},
    {"_name": "rsi", "length": length_choices},
    {"_name": "tsi", "fast": fast_choices, "slow": slow_choices},
]

search_space = {
    "lr":              {"_type": "choice", "_value": [.01, .005, .001, .0005]},
    "rnn_type":        {"_type": "choice", "_value": ['rnn', 'gru']},
    "rnn_hidden_size": {"_type": "choice", "_value": [8, 16, 24]},
    "ind_hidden_size": {"_type": "choice", "_value": [1, 2, 4]},
    "des_size":        {"_type": "choice", "_value": [2, 4, 8, 16]},
    "ind1":            {"_type": "choice", "_value": ind_choices},
    "ind2":            {"_type": "choice", "_value": ind_choices},
}

max_trials = 1_000

# Search Configuration
search = Experiment('local')
# Search Name
search.config.experiment_name = 'Alg Trader'
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
