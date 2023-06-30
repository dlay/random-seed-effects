from pathlib import Path

import numpy as np
import pandas as pd
import pickle as pkl


def collect_evaluation(data_set_names, recommenders):
    scores = {"data_set_name": [], "recommender": [], "shuffle_seed": [], "topn_score": [] ,"precision": [], "ndcg": []}
    for data_set_name in data_set_names:
        for recommender in recommenders:
            fold_evals = {}
            for file in Path(f"./data/{data_set_name}/evaluations_{recommender}").iterdir():
                test_fold, shuffle_seed, topn_score = file.name.split(".")[0].split("_")
                test_fold = int(test_fold)
                if test_fold != 0:
                    continue
                shuffle_seed = int(shuffle_seed)
                if shuffle_seed not in fold_evals:
                    fold_evals[shuffle_seed] = {}
                if topn_score not in fold_evals[shuffle_seed]:
                    fold_evals[shuffle_seed][topn_score] = []
                fold_evals[shuffle_seed][topn_score].append(pd.read_csv(file, header=0, sep=","))
            for shuffle_seed in fold_evals:
                for topn_score in fold_evals[shuffle_seed]:
                    combined_folds = pd.concat(fold_evals[shuffle_seed][topn_score])
                    scores["data_set_name"].append(data_set_name)
                    scores["recommender"].append(recommender)
                    scores["shuffle_seed"].append(shuffle_seed)
                    scores["topn_score"].append(topn_score)
                    scores["precision"].append(combined_folds["precision"].mean())
                    scores["ndcg"].append(combined_folds["ndcg"].mean())

    print()

    return
