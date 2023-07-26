import argparse
from pathlib import Path
import binpickle
import numpy as np
import pandas as pd
import pickle as pkl
from static import *


def make_predictions(data_set_name, prune_technique, split_technique, test_fold, shuffle_seed, recommender,
                     recommender_seed, num_batches, run_batch):
    # get test data
    test_data = pd.read_csv(
        f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}/"
        f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_{SPLIT_FILE}", header=0, sep=",")
    users = test_data["user"].unique()
    # split users into batches
    user_batches = np.array_split(users, num_batches)

    # load recommender
    base_path_recommender = f"./{DATA_FOLDER}/{data_set_name}/{RECOMMENDER_FOLDER}_{recommender}"
    recommender_alg = binpickle.load(f"{base_path_recommender}/"
                                     f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
                                     f"{recommender_seed}_{RECOMMENDER_FILE}")

    # make predictions
    recommendations = {user: recommender_alg.recommend(user, n=20) for user in user_batches[run_batch]}

    # save predictions to file
    base_path_predictions = f"./{DATA_FOLDER}/{data_set_name}/{PREDICTION_FOLDER}_{recommender}"
    Path(base_path_predictions).mkdir(parents=True, exist_ok=True)
    pkl.dump(recommendations, open(f"{base_path_predictions}/"
                                   f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
                                   f"{recommender_seed}_{num_batches}_{run_batch}_{PREDICTION_FILE}", "wb"))
    print(f"Predictions generated and saved.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects make predictions!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    parser.add_argument('--split_technique', dest='split_technique', type=str, required=True)
    parser.add_argument('--test_fold', dest='test_fold', type=int, required=True)
    parser.add_argument('--shuffle_seed', dest='shuffle_seed', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--recommender_seeding', dest='recommender_seeding', type=str, required=True)
    parser.add_argument('--num_batches', dest='num_batches', type=int, required=True)
    parser.add_argument('--run_batch', dest='run_batch', type=int, required=True)
    args = parser.parse_args()

    print("Making predictions with arguments: ", args.__dict__)
    make_predictions(args.data_set_name, args.prune_technique, args.split_technique, args.test_fold,
                     args.shuffle_seed, args.recommender, args.recommender_seeding, args.num_batches, args.run_batch)
