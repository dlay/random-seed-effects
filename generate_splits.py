import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_splits(data_set_name, split_technique, num_folds):
    # load the data
    data = pd.read_csv(f"./data/{data_set_name}/pruned/pruned.csv", header=0, sep=",", index_col=0)

    shuffle_seed = np.random.randint(0, np.iinfo(np.int32).max)
    data = data.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

    if split_technique == "weak_generalization":
        # regular cross validation splits
        splits = np.array_split(data, num_folds)
    else:
        raise ValueError("Split technique not recognized.")

    # write data to file
    base_path_split = f"./data/{data_set_name}/split"
    Path(base_path_split).mkdir(exist_ok=True)
    for i, split in enumerate(splits):
        split = split.sample(frac=1, random_state=42).reset_index(drop=True)
        split.to_csv(f"{base_path_split}/split_{i}_{shuffle_seed}.csv")
    print(f"Written split data set to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects generate splits!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--split_technique', dest='split_technique', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)

    args = parser.parse_args()

    print("Generating splits with arguments: ", args.__dict__)
    generate_splits(args.data_set_name, args.split_technique, args.num_folds)
