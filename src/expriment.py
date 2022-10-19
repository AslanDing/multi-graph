import nni
from nni.experiment import Experiment
search_space = {
    "batch_size": {"_type":"choice", "_value": [32,64,128,256]},
    "learning_rate":{"_type":"uniform","_value":[0.0005,0.0015]}
}
experiment = Experiment('local')
experiment.config.trial_command = 'python3 train_pubmed_nni.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 16
experiment.config.trial_concurrency = 1
experiment.run(8080)
input()
#experiment.stop()
