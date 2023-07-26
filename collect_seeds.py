import json
from pathlib import Path

data_sets = ["adressa", "cds-and-vinyl", "gowalla", "hetrec-lastfm", "movielens-1m", "musical-instruments",
             "retailrocket", "video-games", "yelp"]

recommenders = ["implicit-mf", "item-knn", "popularity"]

seeds = {}
for data_set in data_sets:
    seeds[data_set] = {}
    data_set_seeds = list(set([file.name.split("_")[1] for file in Path(f"./data/{data_set}/split").iterdir()]))
    for seed in data_set_seeds:
        seeds[data_set][seed] = {}
        for recommender in recommenders:
            seeds[data_set][seed][recommender] = {}
            for file in Path(f"./data/{data_set}/recommender_{recommender}").iterdir():
                if seed in file.name:
                    seeds[data_set][seed][recommender][file.name.split("_")[0]] = next(
                        line.strip() for line in open(file, 'r'))

json.dump(seeds, open(f"project_seeds.txt", "w"))

# seeds = json.loads(open(f"project_seeds.txt", "r").read())
