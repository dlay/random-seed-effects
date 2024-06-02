import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from static import *


def generate_splits(data_set_name, prune_technique, split_technique, num_folds, reproducibility_seed):
    # load the data
    data = pd.read_csv(f"./{DATA_FOLDER}/{data_set_name}/{PRUNE_FOLDER}/{prune_technique}_{PRUNE_FILE}", header=0,
                       sep=",")

    # generate shuffle seed and shuffle data
    if reproducibility_seed == -1:
        shuffle_seed = np.random.randint(0, np.iinfo(np.int32).max)
    else:
        shuffle_seed = reproducibility_seed
    data = data.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

    if split_technique == "weak-generalization":
        # regular five-fold cross validation splits
        splits = np.array_split(data, num_folds)
    else:
        raise ValueError("Split technique not recognized.")

    print(f"Split data with technique {split_technique}.")

    # write data to file
    base_path_split = f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}"
    Path(base_path_split).mkdir(exist_ok=True)
    for split_index, split in enumerate(splits):
        split.to_csv(f"{base_path_split}/{split_index}_{shuffle_seed}_{prune_technique}_{split_technique}_{SPLIT_FILE}",
                     index=False)
        stack = np.vstack((np.delete(np.concatenate(splits), split_index, axis=0), splits[split_index]))
        df = pd.DataFrame(stack)
        path = f"{base_path_split}/{split_index}_{shuffle_seed}_{prune_technique}_{split_technique}_split"
        if not os.path.exists(path):
            os.mkdir(path)
        df.to_csv(f"{path}/{split_index}_{shuffle_seed}_{prune_technique}_{split_technique}_split.inter", sep="\t", index=False, header=["user:token", "item:token"])
    print(f"Written split data set to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects generate splits!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    parser.add_argument('--split_technique', dest='split_technique', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--reproducibility_seed', dest='reproducibility_seed', type=int, required=True)

    args = parser.parse_args()

    print("Generating splits with arguments: ", args.__dict__)
    generate_splits(args.data_set_name, args.prune_technique, args.split_technique, args.num_folds,
                    args.reproducibility_seed)
