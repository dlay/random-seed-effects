import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl


def plot_results():
    topn_limiter = [1, 5, 10]
    report = pkl.load(open("evaluation_report.pkl", "rb"))
    for data_set_name in report.keys():
        for prune_technique in report[data_set_name].keys():
            for split_technique in report[data_set_name][prune_technique].keys():
                for recommender in report[data_set_name][prune_technique][split_technique].keys():
                    plot_rows = []
                    for recommender_seed in report[data_set_name][prune_technique][split_technique][recommender].keys():
                        for shuffle_seed in report[data_set_name][prune_technique][split_technique][recommender][recommender_seed].keys():
                            aggregated_results = {}
                            for test_fold in report[data_set_name][prune_technique][split_technique][recommender][recommender_seed][shuffle_seed].keys():
                                for run_batch in report[data_set_name][prune_technique][split_technique][recommender][recommender_seed][shuffle_seed][test_fold].keys():
                                    for topn_score in report[data_set_name][prune_technique][split_technique][recommender][recommender_seed][shuffle_seed][test_fold][run_batch].keys():
                                        if topn_score in topn_limiter:
                                            if test_fold not in aggregated_results:
                                                aggregated_results[test_fold] = {}
                                            if topn_score not in aggregated_results[test_fold]:
                                                aggregated_results[test_fold][topn_score] = []
                                            aggregated_results[test_fold][topn_score].append(report[data_set_name][prune_technique][split_technique][recommender][recommender_seed][shuffle_seed][test_fold][run_batch][topn_score])
                            for test_fold in aggregated_results.keys():
                                for topn_score in aggregated_results[test_fold].keys():
                                    aggregated_results[test_fold][topn_score] = pd.DataFrame(aggregated_results[test_fold][topn_score])
                            topn_scores = list(aggregated_results[0].keys())
                            for topn_score in topn_scores:
                                holdout_precision = aggregated_results[0][topn_score]["precision"].mean()
                                holdout_ndcg = aggregated_results[0][topn_score]["ndcg"].mean()
                                cv_precision = np.array([aggregated_results[fold][topn_score]["precision"].mean() for fold in aggregated_results.keys()]).mean()
                                cv_ndcg = np.array([aggregated_results[fold][topn_score]["ndcg"].mean() for fold in aggregated_results.keys()]).mean()
                                new_row = {"recommender_seed": recommender_seed, "shuffle_seed": shuffle_seed,
                                           "topn_score": topn_score, "validation_type": "holdout",
                                           "metric": "Precision", "metric_value": holdout_precision}
                                plot_rows.append(new_row)
                                new_row = {"recommender_seed": recommender_seed, "shuffle_seed": shuffle_seed,
                                           "topn_score": topn_score, "validation_type": "holdout",
                                           "metric": "nDCG", "metric_value": holdout_ndcg}
                                plot_rows.append(new_row)
                                new_row = {"recommender_seed": recommender_seed, "shuffle_seed": shuffle_seed,
                                           "topn_score": topn_score, "validation_type": "cross-validation",
                                           "metric": "Precision", "metric_value": cv_precision}
                                plot_rows.append(new_row)
                                new_row = {"recommender_seed": recommender_seed, "shuffle_seed": shuffle_seed,
                                           "topn_score": topn_score, "validation_type": "cross-validation",
                                           "metric": "nDCG", "metric_value": cv_ndcg}
                                plot_rows.append(new_row)
                    plot_table = pd.DataFrame(plot_rows)
                    plot_table.rename(columns={'recommender_seed': "Training Seed", 'shuffle_seed': "Data Shuffle Seed",
                                               'topn_score': "k", 'validation_type': "Validation",
                                               'metric': "Metric", 'metric_value': "Metric Value"}, inplace=True)
                    stat_table = pd.DataFrame(columns=["k", "Validation", "Metric", "Mean", "Std", "Var", "Range"])
                    for recommender_seed in plot_table["Training Seed"].unique():
                        for k in plot_table["k"].unique():
                            for validation in plot_table["Validation"].unique():
                                for metric in plot_table["Metric"].unique():
                                    relevant_data = plot_table[(plot_table["Training Seed"] == recommender_seed) & (plot_table["k"] == k) & (plot_table["Validation"] == validation) & (plot_table["Metric"] == metric)]
                                    stat_table = stat_table.append({"k": k,
                                                                    "Validation": validation,
                                                                    "Metric": metric,
                                                                    "Mean": relevant_data["Metric Value"].mean(),
                                                                    "Std": relevant_data["Metric Value"].std(),
                                                                    "Var": relevant_data["Metric Value"].var(),
                                                                    "Range": relevant_data["Metric Value"].max()-relevant_data["Metric Value"].min()}, ignore_index=True)

                                    print(f"{recommender_seed} - {k} - {validation} - {metric}")
                                    print(relevant_data["Metric Value"].std())
                                    print(relevant_data["Metric Value"].var())
                                    print(relevant_data["Metric Value"].max()-relevant_data["Metric Value"].min())
                                    print()

                    latex_table = stat_table.to_latex(index=False)

                    sns.set(font_scale=.5)
                    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
                    cat = sns.catplot(
                        data=plot_table,
                        x="Metric Value", y="Validation", hue="Data Shuffle Seed", row="k", col="Metric", palette="colorblind", height=1, aspect=3, s=10)
                    #sns.move_legend(cat, "upper right")
                    plt.subplots_adjust(top=0.85)
                    # cat.fig.suptitle(f"{data_set_name} - {prune_technique} - {split_technique} - {recommender}")
                    cat.fig.suptitle(f"{data_set_name} - {recommender}")
                    plt.savefig(f'{data_set_name}-{recommender}.pdf', bbox_inches="tight")

                    print()
