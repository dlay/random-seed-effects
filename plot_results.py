import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_results():
    scores = pd.read_csv("results_ho.csv", header=0, sep=",", index_col=0)
    for data_set_name in scores["data_set_name"].unique():
        print(data_set_name)
        for topn_score in scores["topn_score"].unique():
            print(topn_score)
            sns.stripplot(
                data=scores[(scores["data_set_name"] == data_set_name) & (scores["topn_score"] == topn_score)],
                x="precision", y="recommender", hue="shuffle_seed", jitter=True, dodge=True, palette="colorblind")
            plt.show()

    return


if __name__ == "__main__":
    plot_results()
