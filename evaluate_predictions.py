import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pickle as pkl

from static import *
from run_recbole import eval_recbole


def evaluate_predictions(data_set_name, prune_technique, split_technique, test_fold, shuffle_seed, recommender,
                         recommender_seed, reproducibility_seed, num_batches, topn_scores):
    # get test data
    test_data = pd.read_csv(
        f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}/"
        f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_{SPLIT_FILE}", header=0, sep=",")

    evaluation_data = {}
    if recommender == "BPR" or recommender == "MultiVAE":
        # obtain seed for recommender
        if recommender_seed == "random":
            if reproducibility_seed == -1:
                recommender_seed_actual = np.random.randint(0, np.iinfo(np.int32).max)
            else:
                recommender_seed_actual = reproducibility_seed
        elif recommender_seed == "static":
            recommender_seed_actual = 42
        else:
            raise ValueError("Recommender seeding method not recognized.")
        params = {
            "seed": recommender_seed_actual,
            "data_path": f"{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}/",
            "checkpoint_dir": f"{DATA_FOLDER}/{data_set_name}/recommender_{recommender}/"
        }
        file_name = f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_split"
        model_name = f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_{recommender_seed}_recommender.pth"
        predictions = eval_recbole(params, file_name, recommender, model_name)
        evaluation_data[0] = {}
        evaluation_data[0][1] = {"precision": predictions["precision@1"], "ndcg": predictions["ndcg@1"]}
        evaluation_data[0][5] = {"precision": predictions["precision@5"], "ndcg": predictions["ndcg@5"]}
        evaluation_data[0][10] = {"precision": predictions["precision@10"], "ndcg": predictions["ndcg@10"]}
    else:
        for run_batch in range(num_batches):
            # get predictions
            predictions = pkl.load(open(
                f"./{DATA_FOLDER}/{data_set_name}/"
                f"{PREDICTION_FOLDER}_{recommender}/{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
                f"{recommender_seed}_{num_batches}_{run_batch}_{PREDICTION_FILE}", "rb"))
            if run_batch not in evaluation_data:
                evaluation_data[run_batch] = {}

            for topn_score in topn_scores:
                topn_score = int(topn_score)
                # calculate precision and ndcg
                precision_per_user = []
                ndcg_per_user = []
                discounted_gain_per_k = np.array([1 / np.log2(i + 1) for i in range(1, topn_score + 1)])
                idcg = discounted_gain_per_k.sum()
                for user, user_predictions in predictions.items():
                    if user_predictions.shape[0] < topn_score:
                        precision_per_user.append(0)
                        ndcg_per_user.append(0)
                        continue
                    top_k_predictions = user_predictions.values[:topn_score, 0]
                    positive_test_interactions = test_data["item"][test_data["user"] == user].values
                    hits = np.in1d(top_k_predictions, positive_test_interactions)
                    user_precision = sum(hits) / topn_score
                    user_dcg = discounted_gain_per_k[hits].sum()
                    user_ndcg = user_dcg / idcg
                    precision_per_user.append(user_precision)
                    ndcg_per_user.append(user_ndcg)
                total_ndcg = sum(ndcg_per_user) / len(ndcg_per_user)
                total_precision = sum(precision_per_user) / len(precision_per_user)
                evaluation_data[run_batch][topn_score] = {"precision": total_precision, "ndcg": total_ndcg}

    # save results
    base_path_evaluations = f"./{DATA_FOLDER}/{data_set_name}/{EVALUATION_FOLDER}_{recommender}"
    Path(base_path_evaluations).mkdir(parents=True, exist_ok=True)
    topn_scores_string = '-'.join([str(x) for x in topn_scores])
    with open(f"{base_path_evaluations}/"
              f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
              f"{recommender_seed}_{num_batches}_{topn_scores_string}_{EVALUATION_FILE}", "wb") as f:
        pkl.dump(evaluation_data, f)
    print(f"Evaluated predictions and saved results.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects evaluate predictions!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    parser.add_argument('--split_technique', dest='split_technique', type=str, required=True)
    parser.add_argument('--test_fold', dest='test_fold', type=int, required=True)
    parser.add_argument('--shuffle_seed', dest='shuffle_seed', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--recommender_seeding', dest='recommender_seeding', type=str, required=True)
    parser.add_argument('--reproducibility_seed', dest='reproducibility_seed', type=int, required=True)
    parser.add_argument('--num_batches', dest='num_batches', type=int, required=True)
    parser.add_argument('--topn_scores', dest='topn_scores', nargs="+", type=str, required=True)
    args = parser.parse_args()

    print("Evaluating predictions with arguments: ", args.__dict__)
    evaluate_predictions(args.data_set_name, args.prune_technique, args.split_technique, args.test_fold,
                         args.shuffle_seed, args.recommender, args.recommender_seeding, args.reproducibility_seed,
                         args.num_batches, args.topn_scores)
