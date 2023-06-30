import argparse
from pathlib import Path

import binpickle

import pandas as pd
import pickle as pkl

from lenskit.batch import recommend

from static import *


def make_predictions(data_set_name, prune_technique, split_technique, num_folds, test_fold, shuffle_seed, recommender,
                     recommender_seeding):
    # get test data
    test_data = pd.read_csv(
        f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}/"
        f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_{SPLIT_FILE}", header=0, sep=",")
    users = test_data["user"].unique()

    # load recommender
    base_path_recommender = f"./{DATA_FOLDER}/{data_set_name}/{RECOMMENDER_FOLDER}_{recommender}"
    recommender_alg = binpickle.load(f"{base_path_recommender}/"
                                     f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
                                     f"{recommender_seeding}_{RECOMMENDER_FILE}")

    # make predictions
    # recommendations = {user: recommender_alg.recommend(user, n=20) for user in users}
    recommendations = recommend(algo=recommender_alg, users=users, n=20, n_jobs=8)

    # save predictions to file
    base_path_predictions = f"./{DATA_FOLDER}/{data_set_name}/{PREDICTION_FOLDER}_{recommender}"
    Path(base_path_predictions).mkdir(parents=True, exist_ok=True)
    # pkl.dump(recommendations, open(f"{base_path_predictions}/{test_fold}_{shuffle_seed}.pkl", "wb"))
    recommendations.to_csv(f"{base_path_predictions}/{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
                           f"{recommender_seeding}_{PREDICTION_FILE}", index=False)
    print(f"Predictions generated and saved.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scoring optimizer make predictions!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    parser.add_argument('--split_technique', dest='split_technique', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--test_fold', dest='test_fold', type=int, required=True)
    parser.add_argument('--shuffle_seed', dest='shuffle_seed', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--recommender_seeding', dest='recommender_seeding', type=str, required=True)
    args = parser.parse_args()

    print("Making predictions with arguments: ", args.__dict__)
    make_predictions(args.data_set_name, args.prune_technique, args.split_technique, args.num_folds, args.test_fold,
                     args.shuffle_seed, args.recommender, args.recommender_seeding)
