import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def prune_data(data_set_name, prune_technique):
    # load the data
    data = pd.read_csv(f"./data/{data_set_name}/processed/processed.csv", header=0, sep=",", index_col=0)

    if prune_technique == "five_core":
        # apply five core pruning
        u_cnt = Counter(data["user"])
        i_cnt = Counter(data["item"])
        prune_cnt = 0
        while min(u_cnt.values()) < 5 or min(i_cnt.values()) < 5:
            u_sig = [k for k in u_cnt if (u_cnt[k] >= 5)]
            i_sig = [k for k in i_cnt if (i_cnt[k] >= 5)]
            data = data[data["user"].isin(u_sig)]
            data = data[data["item"].isin(i_sig)]
            u_cnt = Counter(data["user"])
            i_cnt = Counter(data["item"])
            prune_cnt += 1
            print(f"Prune loop: {prune_cnt}")
    else:
        raise ValueError("Prune technique not recognized.")

        # write data to file
    base_path_pruned = f"./data/{data_set_name}/pruned"
    Path(base_path_pruned).mkdir(exist_ok=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data.to_csv(f"{base_path_pruned}/pruned.csv")
    print(f"Written pruned data set to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects prune data!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    args = parser.parse_args()

    print("Pruning original with arguments: ", args.__dict__)
    prune_data(args.data_set_name, args.prune_technique)
