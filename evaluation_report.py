from pathlib import Path
from static import *
import pickle as pkl


def evaluation_report(data_set_names, prune_techniques, split_techniques, num_folds, recommenders, recommender_seeding,
                      num_batches, topn_scores):
    topn_scores_string = '-'.join([str(x) for x in topn_scores])
    topn_scores_string_full = '-'.join([str(x) for x in [1, 2, 3, 4, 5, 8, 10, 15, 20]])
    report_dict = {}
    for data_set_name in data_set_names:
        for prune_technique in prune_techniques:
            for split_technique in split_techniques:
                shuffle_seeds = []
                for file in Path(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}").iterdir():
                    _, file_seed, file_prune_technique, file_split_technique, _ = file.name.split(".")[0].split("_")
                    if file_prune_technique == prune_technique and file_split_technique == split_technique:
                        shuffle_seeds.append(file_seed)
                for recommender in recommenders:
                    for shuffle_seed in shuffle_seeds:
                        for recommender_seed in recommender_seeding:
                            for test_fold in range(num_folds):
                                short_path = f"./{DATA_FOLDER}/{data_set_name}/{EVALUATION_FOLDER}_{recommender}/" \
                                             f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
                                             f"{recommender_seed}_{num_batches}_{topn_scores_string}_" \
                                             f"{EVALUATION_FILE}"
                                long_path = f"./{DATA_FOLDER}/{data_set_name}/{EVALUATION_FOLDER}_{recommender}/" \
                                            f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_" \
                                            f"{recommender_seed}_{num_batches}_{topn_scores_string_full}_" \
                                            f"{EVALUATION_FILE}"
                                if Path(short_path).exists():
                                    evaluation_data = pkl.load(open(short_path, "rb"))
                                elif Path(long_path).exists():
                                    evaluation_data = pkl.load(open(long_path, "rb"))
                                else:
                                    raise ValueError(f"File {short_path} or {long_path} does not exist.")
                                if data_set_name not in report_dict:
                                    report_dict[data_set_name] = {}
                                if prune_technique not in report_dict[data_set_name]:
                                    report_dict[data_set_name][prune_technique] = {}
                                if split_technique not in report_dict[data_set_name][prune_technique]:
                                    report_dict[data_set_name][prune_technique][split_technique] = {}
                                if recommender not in report_dict[data_set_name][prune_technique][split_technique]:
                                    report_dict[data_set_name][prune_technique][split_technique][
                                        recommender] = {}
                                if recommender_seed not in report_dict[data_set_name][prune_technique][split_technique][
                                    recommender]:
                                    report_dict[data_set_name][prune_technique][split_technique][
                                        recommender][recommender_seed] = {}
                                if shuffle_seed not in report_dict[data_set_name][prune_technique][
                                    split_technique][recommender][recommender_seed]:
                                    report_dict[data_set_name][prune_technique][split_technique][
                                        recommender][recommender_seed][shuffle_seed] = {}
                                if test_fold not in report_dict[data_set_name][prune_technique][
                                    split_technique][recommender][recommender_seed][shuffle_seed]:
                                    report_dict[data_set_name][prune_technique][split_technique][
                                        recommender][recommender_seed][shuffle_seed][test_fold] = {}
                                report_dict[data_set_name][prune_technique][split_technique][recommender][
                                    recommender_seed][shuffle_seed][test_fold] = evaluation_data

    with open(f"evaluation_report.pkl", "wb") as file:
        pkl.dump(report_dict, file)
    print("Evaluation report saved.")

    return
