import json
from pathlib import Path
from select_experiment import file
import numpy as np

experiment_settings = json.load(open(f"./experiment_{file}.json"))

seeds = {}
for data_set in experiment_settings["DATA_SET_NAMES"]:
    seeds[data_set] = {}
    data_set_seeds = [str(np.random.randint(0, np.iinfo(np.int32).max)) for i in range(20)]
    for seed in data_set_seeds:
        seeds[data_set][seed] = {}
        for recommender in experiment_settings["RECOMMENDERS"]:
            seeds[data_set][seed][recommender] = {}
            for fold in range(experiment_settings["NUM_FOLDS"]):
                seeds[data_set][seed][recommender][str(fold)] = str(np.random.randint(0, np.iinfo(np.int32).max))

json.dump(seeds, open(f"project_seeds.txt", "w"))