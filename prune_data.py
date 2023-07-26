import argparse
from collections import Counter
from pathlib import Path
import pandas as pd
from static import *


def prune_data(data_set_name, prune_technique):
    # load the data
    data = pd.read_csv(f"./{DATA_FOLDER}/{data_set_name}/{CLEAN_FOLDER}/{CLEAN_FILE}", header=0, sep=",")

    if prune_technique == "five-core":
        # apply five core pruning
        u_cnt, i_cnt = Counter(data["user"]), Counter(data["item"])
        while min(u_cnt.values()) < 5 or min(i_cnt.values()) < 5:
            u_sig = [k for k in u_cnt if (u_cnt[k] >= 5)]
            i_sig = [k for k in i_cnt if (i_cnt[k] >= 5)]
            data = data[data["user"].isin(u_sig)]
            data = data[data["item"].isin(i_sig)]
            u_cnt, i_cnt = Counter(data["user"]), Counter(data["item"])
    elif prune_technique == "five-user-warm-start":
        # apply five user warm start pruning
        u_cnt = Counter(data["user"])
        u_sig = [k for k in u_cnt if (u_cnt[k] >= 5)]
        data = data[data["user"].isin(u_sig)]
    elif prune_technique == "five-item-warm-start":
        # apply five item warm start pruning
        i_cnt = Counter(data["item"])
        i_sig = [k for k in i_cnt if (i_cnt[k] >= 5)]
        data = data[data["item"].isin(i_sig)]
    elif prune_technique == "none":
        # apply no pruning
        pass
    else:
        raise ValueError("Prune technique not recognized.")

    print(f"Pruned data with technique: {prune_technique}.")

    # write data to file
    base_path_pruned = f"./{DATA_FOLDER}/{data_set_name}/{PRUNE_FOLDER}"
    Path(base_path_pruned).mkdir(exist_ok=True)
    data.to_csv(f"{base_path_pruned}/{prune_technique}_{PRUNE_FILE}", index=False)
    print(f"Written pruned data set to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects prune data!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--prune_technique', dest='prune_technique', type=str, required=True)
    args = parser.parse_args()

    print("Pruning data with arguments: ", args.__dict__)
    prune_data(args.data_set_name, args.prune_technique)
