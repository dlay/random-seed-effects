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
from lenskit.algorithms.funksvd import FunkSVD
from static import *
from run_recbole import fit_recbole

def fit_recommender(data_set_name, prune_technique, split_technique, num_folds, test_fold, shuffle_seed, recommender,
                    recommender_seed, reproducibility_seed):
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
    if recommender_seed == "random":
        if reproducibility_seed == -1:
            recommender_seed_actual = np.random.randint(0, np.iinfo(np.int32).max)
        else:
            recommender_seed_actual = reproducibility_seed
    elif recommender_seed == "static":
        recommender_seed_actual = 42
    else:
        raise ValueError("Recommender seeding method not recognized.")

    # select the recommender
    if recommender == "random":
        recommender_alg = Random(rng_spec=recommender_seed_actual)
    elif recommender == "popularity":
        recommender_alg = Recommender.adapt(PopScore())
    elif recommender == "implicit-mf":
        recommender_alg = Recommender.adapt(ImplicitMF(features=100, rng_spec=recommender_seed_actual))
    elif recommender == "user-knn":
        recommender_alg = Recommender.adapt(UserUser(nnbrs=20, feedback='implicit', rng_spec=recommender_seed_actual))
    elif recommender == "item-knn":
        recommender_alg = Recommender.adapt(ItemItem(nnbrs=20, feedback='implicit', rng_spec=recommender_seed_actual))
    elif recommender == "funk-svd":
        recommender_alg = Recommender.adapt(FunkSVD(features=100, random_state=recommender_seed_actual))
    elif recommender == "BPR":
        params = {
            "seed": recommender_seed_actual,
            "data_path": f"{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}/",
            "checkpoint_dir": f"{DATA_FOLDER}/{data_set_name}/recommender_{recommender}/",
            "embedding_size": 100
        }
        file_name = f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_split"
        trainer, dataloader = fit_recbole(params, file_name, recommender)
        trainer.saved_model_file = f"{DATA_FOLDER}/{data_set_name}/recommender_{recommender}/{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_{recommender_seed}_recommender.pth"
        trainer.fit(dataloader, saved=True, show_progress=False)
    elif recommender == "MultiVAE":
        params = {
            "seed": recommender_seed_actual,
            "data_path": f"{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}/",
            "checkpoint_dir": f"{DATA_FOLDER}/{data_set_name}/recommender_{recommender}/",
            "latent_dimension": 128,
            "mlp_hidden_size": [600],
            "dropout_prob": 0.5,
            "anneal_cap": 0.2,
            "total_anneal_steps": 200000
        }
        file_name = f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_split"
        trainer, dataloader = fit_recbole(params, file_name, recommender)
        trainer.saved_model_file = f"{DATA_FOLDER}/{data_set_name}/recommender_{recommender}/{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_{recommender_seed}_recommender.pth"
        trainer.fit(dataloader, saved=True, show_progress=False)
    else:
        raise ValueError("Recommender not supported!")

    if recommender != "BPR" and recommender != "MultiVAE":
        # fit recommender
        recommender_alg.fit(train_data)

        # save recommender to file
        base_path_recommender = f"./{DATA_FOLDER}/{data_set_name}/{RECOMMENDER_FOLDER}_{recommender}"
        Path(base_path_recommender).mkdir(exist_ok=True)
        binpickle.dump(recommender_alg, f"{base_path_recommender}/"
                                        f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
                                        f"{recommender_seed}_{RECOMMENDER_FILE}")
        with open(f"{base_path_recommender}/"
                f"{test_fold}_{shuffle_seed}_{prune_technique}_{split_technique}_"
                f"{recommender_seed}_{RECOMMENDER_SEED_FILE}", "w") as f:
            f.write(f"{recommender_seed_actual}")
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
    parser.add_argument('--reproducibility_seed', dest='reproducibility_seed', type=int, required=True)
    args = parser.parse_args()

    print("Fitting recommender with arguments: ", args.__dict__)
    fit_recommender(args.data_set_name, args.prune_technique, args.split_technique, args.num_folds, args.test_fold,
                    args.shuffle_seed, args.recommender, args.recommender_seeding, args.reproducibility_seed)
