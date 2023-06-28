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
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import CosineRecommender, TFIDFRecommender, BM25Recommender
from scipy.sparse import csr_matrix


def fit_recommender(data_set_name, num_folds, test_fold, shuffle_seed, recommender):
    # get train data
    train_folds = [x for x in range(num_folds) if x != test_fold]
    train_data_dfs = []
    for train_fold in train_folds:
        train_data_dfs.append(
            pd.read_csv(f"./data/{data_set_name}/split/split_{train_fold}_{shuffle_seed}.csv", header=0, sep=",",
                        index_col=0))
    train_data = pd.concat(train_data_dfs, ignore_index=True)

    # fit an implicit recommender
    if recommender == "random":
        recommender_alg = Random(rng_spec=42)
    elif recommender == "popularity":
        recommender_alg = Recommender.adapt(PopScore())
    elif recommender == "implicit-mf":
        recommender_alg = Recommender.adapt(ImplicitMF(features=100, rng_spec=42))
    elif recommender == "user-knn":
        recommender_alg = Recommender.adapt(UserUser(nnbrs=20, feedback='implicit', rng_spec=42))
    elif recommender == "item-knn":
        recommender_alg = Recommender.adapt(ItemItem(nnbrs=20, feedback='implicit', rng_spec=42))
    elif recommender == "alternating-least-squares":
        recommender_alg = AlternatingLeastSquares(factors=100, random_state=42)
    elif recommender == "bayesian-personalized-ranking":
        recommender_alg = BayesianPersonalizedRanking(factors=100, random_state=42)
    elif recommender == "logistic-mf":
        recommender_alg = LogisticMatrixFactorization(factors=100, random_state=42)
    else:
        raise ValueError("Recommender not supported!")

    if recommender in ["random", "popularity", "implicit-mf", "user-knn", "item-knn"]:
        recommender_alg.fit(train_data)
    elif recommender in ["alternating-least-squares", "bayesian-personalized-ranking", "logistic-mf"]:
        matrix = csr_matrix((np.ones(train_data.shape[0]), (train_data["user"].values, train_data["item"].values)))
        recommender_alg.fit(matrix)
    else:
        raise ValueError("Recommender not supported!")

    # save recommender to file
    base_path_recommender = f"./data/{data_set_name}/recommender_{recommender}"
    Path(base_path_recommender).mkdir(exist_ok=True)
    binpickle.dump(recommender_alg, f"{base_path_recommender}/{test_fold}_{shuffle_seed}.bpk")
    print(f"Fitted recommender and saved to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scoring optimizer fit recommender!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--test_fold', dest='test_fold', type=int, required=True)
    parser.add_argument('--shuffle_seed', dest='shuffle_seed', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    args = parser.parse_args()

    print("Fitting recommender with arguments: ", args.__dict__)
    fit_recommender(args.data_set_name, args.num_folds, args.test_fold, args.shuffle_seed, args.recommender)
