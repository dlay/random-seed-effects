import argparse
from pathlib import Path

import binpickle
import numpy as np
import pandas as pd
import pickle as pkl

from scipy.sparse import csr_matrix


def make_predictions(data_set_name, test_fold, shuffle_seed, recommender):
    # get test data
    test_data = pd.read_csv(f"./data/{data_set_name}/split/split_{test_fold}_{shuffle_seed}.csv", header=0, sep=",",
                            index_col=0)
    users = test_data["user"].unique()

    # load recommender
    base_path_recommender = f"./data/{data_set_name}/recommender_{recommender}"
    recommender_alg = binpickle.load(f"{base_path_recommender}/{test_fold}_{shuffle_seed}.bpk")
    if recommender in ["random", "popularity", "implicit-mf", "user-knn", "item-knn"]:
        # make predictions
        recommendations = {user: recommender_alg.recommend(user, n=20) for user in users}
    elif recommender in ["alternating-least-squares", "bayesian-personalized-ranking", "logistic-mf"]:
        matrix = csr_matrix((np.ones(test_data.shape[0]), (test_data["user"].values, test_data["item"].values)))
        recommendations = {}
        for user in users:
            user_recs = recommender_alg.recommend(user, matrix[user], N=20)
            recommendations[user] = pd.DataFrame(dict(item=user_recs[0], score=user_recs[1]))
    else:
        raise ValueError(f"Recommender {recommender} not supported.")

    # save predictions to file
    base_path_predictions = f"./data/{data_set_name}/predictions_{recommender}"
    Path(base_path_predictions).mkdir(parents=True, exist_ok=True)
    pkl.dump(recommendations, open(f"{base_path_predictions}/{test_fold}_{shuffle_seed}.pkl", "wb"))
    print(f"Predictions generated and saved.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scoring optimizer make predictions!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--test_fold', dest='test_fold', type=int, required=True)
    parser.add_argument('--shuffle_seed', dest='shuffle_seed', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    args = parser.parse_args()

    print("Making predictions with arguments: ", args.__dict__)
    make_predictions(args.data_set_name, args.test_fold, args.shuffle_seed, args.recommender)
