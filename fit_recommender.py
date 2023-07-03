import argparse
from pathlib import Path

import binpickle
import numpy as np

import pandas as pd
from lenskit import Recommender
from lenskit.algorithms.basic import Random, PopScore
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem

from static import *


def fit_recommender(data_set_name, prune_technique, split_technique, num_folds, test_fold, shuffle_seed, recommender,
                    recommender_seeding):
    # get train data
    train_folds = [x for x in range(num_folds) if x != test_fold]
    train_data_dfs = []
    for train_fold in train_folds:
        train_data_dfs.append(
            pd.read_csv(f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}/"
                        f"{train_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_{SPLIT_FILE}",
                        header=0, sep=",", ))
    train_data = pd.concat(train_data_dfs, ignore_index=True)

    # obtain seed for recommender
    if recommender_seeding == "random":
        recommender_seed = np.random.randint(0, np.iinfo(np.int32).max)
    elif recommender_seeding == "static":
        recommender_seed = 42
    else:
        raise ValueError("Recommender seeding method not recognized.")

    # select the recommender
    if recommender == "random":
        recommender_alg = Random(rng_spec=recommender_seed)
    elif recommender == "popularity":
        recommender_alg = Recommender.adapt(PopScore())
    elif recommender == "implicit-mf":
        recommender_alg = Recommender.adapt(ImplicitMF(features=100, rng_spec=recommender_seed))
    elif recommender == "user-knn":
        recommender_alg = Recommender.adapt(UserUser(nnbrs=20, feedback='implicit', rng_spec=recommender_seed))
    elif recommender == "item-knn":
        recommender_alg = Recommender.adapt(ItemItem(nnbrs=20, feedback='implicit', rng_spec=recommender_seed))
    else:
        raise ValueError("Recommender not supported!")

    # fit recommender
    recommender_alg.fit(train_data)

    # save recommender to file
    base_path_recommender = f"./{DATA_FOLDER}/{data_set_name}/{RECOMMENDER_FOLDER}_{recommender}"
    Path(base_path_recommender).mkdir(exist_ok=True)
    binpickle.dump(recommender_alg, f"{base_path_recommender}/"
                                    f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
                                    f"{recommender_seeding}_{RECOMMENDER_FILE}")
    with open(f"{base_path_recommender}/"
              f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
              f"{recommender_seeding}_{RECOMMENDER_SEED_FILE}", "w") as f:
        f.write(f"{recommender_seed}")
    print(f"Fitted recommender and saved to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects fit recommender!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    parser.add_argument('--split_technique', dest='split_technique', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--test_fold', dest='test_fold', type=int, required=True)
    parser.add_argument('--shuffle_seed', dest='shuffle_seed', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--recommender_seeding', dest='recommender_seeding', type=str, required=True)
    args = parser.parse_args()

    print("Fitting recommender with arguments: ", args.__dict__)
    fit_recommender(args.data_set_name, args.prune_technique, args.split_technique, args.num_folds, args.test_fold,
                    args.shuffle_seed, args.recommender, args.recommender_seeding)
