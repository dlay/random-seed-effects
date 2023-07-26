import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from static import *


def clean_data(data_set_name):
    # the path of the original data
    base_path_original = f"./{DATA_FOLDER}/{data_set_name}/{ORIGINAL_FOLDER}"

    # load the data into a dataframe
    if data_set_name == "adressa":
        dfs = []
        for file in Path(f"{base_path_original}").iterdir():
            with open(file, 'r') as f:
                file_data = []
                for line in f.readlines():
                    line_data = json.loads(line)
                    if "id" in line_data and "userId" in line_data:
                        file_data.append([line_data["userId"], line_data["id"]])
                dfs.append(pd.DataFrame(file_data, columns=["user", "item"]))
        data = pd.concat(dfs).copy()
    elif data_set_name == "cds-and-vinyl" or data_set_name == "musical-instruments" or data_set_name == "video-games":
        data = pd.read_json(f"{base_path_original}/amazon.json", lines=True,
                            dtype={
                                'reviewerID': str,
                                'asin': str,
                                'overall': np.float64,
                                'unixReviewTime': np.float64
                            })[['reviewerID', 'asin', 'overall']]
        data.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating'}, inplace=True)
        data = data[data["rating"] > 3][["user", "item"]]
    elif data_set_name == "gowalla":
        data = pd.read_csv(f"{base_path_original}/Gowalla_totalCheckins.txt",
                           names=["user", "check-in time", "latitude", "longitude", "item"],
                           usecols=["user", "item"], header=None, sep="\t")
    elif data_set_name == "hetrec-lastfm":
        data = pd.read_csv(f"{base_path_original}/user_artists.dat", names=["user", "item", "weight"],
                           usecols=["user", "item"], header=0, sep="\t")
    elif data_set_name == "movielens-1m":
        data = pd.read_csv(f"{base_path_original}/ratings.dat", header=None, sep="::",
                           names=["user", "item", "rating", "timestamp"], usecols=["user", "item", "rating"])
        data = data[data["rating"] > 3][["user", "item"]]
    elif data_set_name == "retailrocket":
        data = pd.read_csv(f"{base_path_original}/events.csv", usecols=["visitorid", "itemid", "event"], header=0,
                           sep=",")
        data.rename(columns={"visitorid": "user", "itemid": "item"}, inplace=True)
        data = data[data["event"] == "view"][["user", "item"]].copy()
    elif data_set_name == "yelp":
        final_dict = {'user': [], 'item': []}
        with open(f"{base_path_original}/yelp_academic_dataset_review.json", encoding="utf8") as file:
            for line in file:
                dic = eval(line)
                if all(k in dic for k in ("user_id", "business_id")):
                    final_dict['user'].append(dic['user_id'])
                    final_dict['item'].append(dic['business_id'])
        data = pd.DataFrame.from_dict(final_dict)
    else:
        raise ValueError(f"Unknown data set name {data_set_name}.")

    # remove duplicates
    data.drop_duplicates(inplace=True)

    # map user and item to integers
    for col in ["user", "item"]:
        unique_ids = {key: value for value, key in enumerate(data[col].unique())}
        data[col].update(data[col].map(unique_ids))

    print("Dropped duplicates and mapped user and item to integers.")

    # write data to file
    base_path_cleaned = f"./{DATA_FOLDER}/{data_set_name}/{CLEAN_FOLDER}"
    Path(base_path_cleaned).mkdir(exist_ok=True)
    data.to_csv(f"{base_path_cleaned}/{CLEAN_FILE}", index=False)
    print(f"Written cleaned data set to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects clean data!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    args = parser.parse_args()

    print("Pruning original with arguments: ", args.__dict__)
    clean_data(args.data_set_name)
