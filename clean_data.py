import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean_data(data_set_name):
    base_path_original = f"./data/{data_set_name}/original"

    if not Path(f"{base_path_processed}/processed.csv").exists():
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
        elif data_set_name == "yelp":
            final_dict = {'user': [], 'item': []}
            with open(f"{base_path_original}/yelp_academic_dataset_review.json", encoding="utf8") as file:
                for line in file:
                    dic = eval(line)
                    if all(k in dic for k in ("user_id", "business_id")):
                        final_dict['user'].append(dic['user_id'])
                        final_dict['item'].append(dic['business_id'])
            data = pd.DataFrame.from_dict(final_dict)
        elif data_set_name == "citeulike-a":
            u_i_pairs = []
            with open(f"{base_path_original}/users.dat", "r") as f:
                for user, line in enumerate(f.readlines()):
                    item_cnt = line.strip("\n").split(" ")[0]  # First element is a count
                    items = line.strip("\n").split(" ")[1:]
                    assert len(items) == int(item_cnt)

                    for item in items:
                        # Make sure the identifiers are correct.
                        assert item.isdecimal()
                        u_i_pairs.append((user, int(item)))

            # Rename columns to default ones ?
            data = pd.DataFrame(
                u_i_pairs,
                columns=["user", "item"],
                dtype=np.int64,
            )
        elif data_set_name == "cosmetics-shop":
            dfs = []
            for file in Path(f"{base_path_original}").iterdir():
                with open(file, 'r') as f:
                    df = pd.read_csv(f, usecols=["user_id", "product_id", "event_type"], header=0, sep=",")
                    df.rename(columns={"user_id": "user", "product_id": "item"}, inplace=True)
                    df = df[df["event_type"] == "view"][["user", "item"]].copy()
                    dfs.append(df)
            data = pd.concat(dfs).copy()
        elif data_set_name == "globo":
            dfs = []
            for item in Path(f"{base_path_original}").iterdir():
                with open(item, 'r') as f:
                    df = pd.read_csv(f, usecols=["user_id", "click_article_id"], sep=",")
                    df.rename(columns={"user_id": "user", "click_article_id": "item"}, inplace=True)
                    if df.shape[0] == 0:
                        continue
                    else:
                        dfs.append(df)
            data = pd.concat(dfs).copy()
        elif data_set_name == "yoochoose":
            data = pd.read_csv(f"{base_path_original}/yoochoose-clicks.dat",
                               names=["user", "timestamp", "item", "category"],
                               dtype={"user": np.int64, "item": np.int64, }, usecols=["user", "item"], header=None)
        elif data_set_name == "retailrocket":
            data = pd.read_csv(f"{base_path_original}/events.csv", usecols=["visitorid", "itemid", "event"], header=0,
                               sep=",")
            data.rename(columns={"visitorid": "user", "itemid": "item"}, inplace=True)
            data = data[data["event"] == "view"][["user", "item"]].copy()
        elif data_set_name == "hetrec-lastfm":
            data = pd.read_csv(f"{base_path_original}/user_artists.dat", names=["user", "item", "weight"],
                               usecols=["user", "item"], header=0, sep="\t")
        elif data_set_name == "gowalla":
            data = pd.read_csv(f"{base_path_original}/Gowalla_totalCheckins.txt",
                               names=["user", "check-in time", "latitude", "longitude", "item"],
                               usecols=["user", "item"], header=None, sep="\t")
        elif data_set_name == "sketchfab":
            user = []
            model = []
            with open(f"{base_path_original}/model_likes_anon.psv", "rb") as file:
                file.readline()
                for line in file:
                    pipe_split = line.split(b"|")
                    if len(pipe_split) >= 3:
                        user.append(pipe_split[-2].decode("utf-8"))
                        model.append(pipe_split[-1].decode("utf-8"))
            data = pd.DataFrame({"user": user, "item": model})
        elif data_set_name == "spotify-playlists":
            user = []
            track = []
            with open(f"{base_path_original}/spotify_dataset.csv", "rb") as f:
                f.readline()
                for idx, line in enumerate(f):
                    line_split = line.split(b'","')
                    user.append(line_split[0][1:].decode("utf-8"))
                    track.append(line_split[2].decode("utf-8"))
            data = pd.DataFrame({"user": user, "item": track})
        elif data_set_name == "nowplaying":
            data = pd.read_csv(f"{base_path_original}/user_track_hashtag_timestamp.csv", header=0, sep=",",
                               usecols=["user_id", "track_id"])
            data.rename(columns={"user_id": "user", "track_id": "item"}, inplace=True)
        else:
            raise ValueError(f"Unknown data set name {data_set_name}.")

        # remove duplicates
        data.drop_duplicates(inplace=True)

        # map user and item to integers
        for col in ["user", "item"]:
            unique_ids = {key: value for value, key in enumerate(data[col].unique())}
            data[col].update(data[col].map(unique_ids))

        # write data to file
        base_path_processed = f"./data/{data_set_name}/processed"
        Path(base_path_processed).mkdir(exist_ok=True)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        data.to_csv(f"{base_path_processed}/processed.csv")
        print(f"Written processed data set to file.")

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Random Seed Effects clean data!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    args = parser.parse_args()

    print("Pruning original with arguments: ", args.__dict__)
    clean_data(args.data_set_name)
