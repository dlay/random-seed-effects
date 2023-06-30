import json
import subprocess
from pathlib import Path
from select_experiment import experiment_file, stage
from collect_evaluation import collect_evaluation

from static import *


def execute_clean_data(data_set_names):
    for data_set_name in data_set_names:
        base_path = f"./{DATA_FOLDER}/{data_set_name}/{CLEAN_FOLDER}/{CLEAN_FILE}"
        if not Path(base_path).exists():
            subprocess.run(["py", "-3.9", "clean_data.py", "--data_set_name", f"{data_set_name}"])


def execute_prune_data(data_set_names, prune_techniques):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            base_path = f"./{DATA_FOLDER}/{data_set_name}/{PRUNE_FOLDER}/{prune_technique}_{PRUNE_FILE}"
            if not Path(base_path).exists():
                subprocess.run(
                    ["py", "-3.9", "prune_data.py", "--data_set_name", f"{data_set_name}", "--prune_technique",
                     f"{prune_technique}"])


def execute_generate_splits(data_set_names, prune_techniques, split_techniques, num_folds):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                subprocess.run(
                    ["py", "-3.9", "generate_splits.py", "--data_set_name", f"{data_set_name}", "--prune_technique",
                     f"{prune_technique}", "--split_technique", f"{split_technique}", "--num_folds", f"{num_folds}"])


def execute_fit_recommender(data_set_names, prune_techniques, split_techniques, num_folds, recommenders,
                            recommender_seeding):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                shuffle_seeds = []
                for file in Path(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}").iterdir():
                    _, file_seed, file_prune_technique, file_split_technique, _ = file.name.split(".")[0].split("_")
                    if file_prune_technique == prune_technique and file_split_technique == split_technique:
                        shuffle_seeds.append(file_seed)
                shuffle_seeds = list(set(shuffle_seeds))
                for recommender in recommenders:
                    for shuffle_seed in shuffle_seeds:
                        for recommender_seed in recommender_seeding:
                            for test_fold in range(num_folds):
                                base_path = f"./{DATA_FOLDER}/{data_set_name}/{RECOMMENDER_FOLDER}_{recommender}/" \
                                            f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
                                            f"{recommender_seed}_{RECOMMENDER_FILE}"
                                if not Path(base_path).exists():
                                    subprocess.run(
                                        ["py", "-3.9", "fit_recommender.py", "--data_set_name",
                                         f"{data_set_name}", "--prune_technique", f"{prune_technique}",
                                         "--split_technique", f"{split_technique}", "--num_folds", f"{num_folds}",
                                         "--test_fold", f"{test_fold}", "--shuffle_seed", f"{shuffle_seed}",
                                         "--recommender", f"{recommender}", "--recommender_seeding",
                                         f"{recommender_seed}"])


def execute_make_predictions(data_set_names, prune_techniques, split_techniques, num_folds, recommenders,
                             recommender_seeding):
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                shuffle_seeds = []
                for file in Path(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}").iterdir():
                    _, file_seed, file_prune_technique, file_split_technique, _ = file.name.split(".")[0].split("_")
                    if file_prune_technique == prune_technique and file_split_technique == split_technique:
                        shuffle_seeds.append(file_seed)
                shuffle_seeds = list(set(shuffle_seeds))
                for recommender in recommenders:
                    for shuffle_seed in shuffle_seeds:
                        for recommender_seed in recommender_seeding:
                            for test_fold in range(num_folds):
                                base_path = f"./{DATA_FOLDER}/{data_set_name}/{PREDICTION_FOLDER}_{recommender}/" \
                                            f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
                                            f"{recommender_seed}_{PREDICTION_FILE}"
                                if not Path(base_path).exists():
                                    subprocess.run(
                                        ["py", "-3.9", "make_predictions.py", "--data_set_name",
                                         f"{data_set_name}", "--prune_technique", f"{prune_technique}",
                                         "--split_technique", f"{split_technique}", "--num_folds", f"{num_folds}",
                                         "--test_fold", f"{test_fold}", "--shuffle_seed", f"{shuffle_seed}",
                                         "--recommender", f"{recommender}", "--recommender_seeding",
                                         f"{recommender_seed}"])


def execute_evaluate_predictions(data_set_names, num_folds, recommenders, topn_scores):
    for data_set_name in data_set_names:
        shuffle_seeds = []
        for file in Path(f"./data/{data_set_name}/split").iterdir():
            shuffle_seeds.append(file.name.split(".")[0].split("_")[-1])
        shuffle_seeds = list(set(shuffle_seeds))
        for recommender in recommenders:
            for shuffle_seed in shuffle_seeds:
                for i in range(num_folds):
                    for topn_score in topn_scores:
                        base_path = f"./data/{data_set_name}/evaluations_{recommender}/" \
                                    f"{i}_{shuffle_seed}_{topn_score}.csv"
                        if not Path(base_path).exists():
                            subprocess.run(
                                ["py", "-3.9", "evaluate_predictions.py", "--data_set_name", f"{data_set_name}",
                                 "--test_fold", f"{i}", "--shuffle_seed", f"{shuffle_seed}",
                                 "--recommender", f"{recommender}", "--topn_score", f"{topn_score}"])


def execute_collect_evaluation(data_set_names, recommenders):
    collect_evaluation(data_set_names, recommenders)


experiment_settings = json.load(open(f"./experiment_{experiment_file}.json"))
if stage == 0:
    execute_clean_data(experiment_settings["DATA_SET_NAMES"])
elif stage == 1:
    execute_prune_data(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"])
elif stage == 2:
    execute_generate_splits(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                            experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"])
elif stage == 3:
    execute_fit_recommender(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                            experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                            experiment_settings["RECOMMENDERS"], experiment_settings["RECOMMENDER_SEEDING"])
elif stage == 4:
    execute_make_predictions(experiment_settings["DATA_SET_NAMES"], experiment_settings["PRUNE_TECHNIQUES"],
                             experiment_settings["SPLIT_TECHNIQUES"], experiment_settings["NUM_FOLDS"],
                             experiment_settings["RECOMMENDERS"], experiment_settings["RECOMMENDER_SEEDING"])
elif stage == 5:
    execute_evaluate_predictions(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                                 experiment_settings["RECOMMENDERS"], experiment_settings["TOPN_SCORES"])
elif stage == 6:
    execute_collect_evaluation(experiment_settings["DATA_SET_NAMES"], experiment_settings["RECOMMENDERS"])

else:
    print("No valid stage selected!")