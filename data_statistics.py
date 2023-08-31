import json
import pandas as pd
from select_experiment import file
from static import *

experiment_settings = json.load(open(f"./experiment_{file}.json"))

info_df = pd.DataFrame(
    columns=["#Interactions", "#Users", "#Items", "Avg.#Int. per user",
             "Avg.#Int. per item", "Sparsity"])

for data_set in experiment_settings["DATA_SET_NAMES"]:
    data = pd.read_csv(f"./{DATA_FOLDER}/{data_set}/{PRUNE_FOLDER}/{PRUNE_FILE}", sep=",", header=0)
    users = data["user"].unique()
    items = data["item"].unique()
    interactions = data[["user", "item"]].values
    number_of_users = len(users)
    number_of_items = len(items)
    number_of_interactions = len(interactions)
    sparsity = 1 - (number_of_interactions / (number_of_users * number_of_items))
    average_interactions_per_user = number_of_interactions / number_of_users
    average_interactions_per_item = number_of_interactions / number_of_items

    # print number of users, items, interactions and sparsity
    print("-" * 50)
    print(f"Data set: {data_set}")
    print(f"Number of interactions: {number_of_interactions}")
    print(f"Number of users: {number_of_users}")
    print(f"Number of items: {number_of_items}")
    print(f"Average ratings per user: {average_interactions_per_user}")
    print(f"Average ratings per item: {average_interactions_per_item}")
    print(f"Sparsity: {sparsity}")

    info_df.loc[data_set] = [f'{number_of_interactions:,}', f'{number_of_users:,}', f'{number_of_items:,}',
                             f'{round(average_interactions_per_user, 2):,}',
                             f'{round(average_interactions_per_item, 2):,}', f"{round(sparsity * 100, 2)}%"]

    print(info_df.to_latex())
