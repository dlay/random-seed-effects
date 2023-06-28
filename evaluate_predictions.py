import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pickle as pkl


def evaluate_predictions(data_set_name, test_fold, shuffle_seed, recommender, topn_score):
    test_data = pd.read_csv(f"./data/{data_set_name}/split/split_{test_fold}_{shuffle_seed}.csv", header=0, sep=",",
                            index_col=0)
    predictions = pkl.load(open(f"./data/{data_set_name}/predictions_{recommender}/{test_fold}_{shuffle_seed}.pkl",
                                "rb"))

    precision_per_user = []
    ndcg_per_user = []
    discounted_gain_per_k = np.array([1 / np.log2(i + 1) for i in range(1, topn_score + 1)])
    idcg = discounted_gain_per_k.sum()
    for user, predictions in predictions.items():
        if predictions.shape[0] < topn_score:
            precision_per_user.append(0)
            ndcg_per_user.append(0)
            continue
        top_k_predictions = predictions.values[:topn_score, 0]
        positive_test_interactions = test_data["item"][test_data["user"] == user].values
        hits = np.in1d(top_k_predictions, positive_test_interactions)
        user_precision = sum(hits) / topn_score
        user_dcg = discounted_gain_per_k[hits].sum()
        user_ndcg = user_dcg / idcg
        precision_per_user.append(user_precision)
        ndcg_per_user.append(user_ndcg)
    total_ndcg = sum(ndcg_per_user) / len(ndcg_per_user)
    total_precision = sum(precision_per_user) / len(precision_per_user)

    base_path_evaluations = f"./data/{data_set_name}/evaluations_{recommender}"
    Path(base_path_evaluations).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"precision": [total_precision], "ndcg": [total_ndcg]}).to_csv(
        f"./data/{data_set_name}/evaluations_{recommender}/{test_fold}_{shuffle_seed}_{topn_score}.csv", header=True,
        index=False)
    print(f"Evaluated predictions and saved results.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scoring optimizer evaluate predictions!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--test_fold', dest='test_fold', type=int, required=True)
    parser.add_argument('--shuffle_seed', dest='shuffle_seed', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--topn_score', dest='topn_score', type=int, required=True)
    args = parser.parse_args()

    print("Evaluating predictions with arguments: ", args.__dict__)
    evaluate_predictions(args.data_set_name, args.test_fold, args.shuffle_seed, args.recommender, args.topn_score)
