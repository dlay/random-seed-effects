from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl
from scipy.stats import wilcoxon


def plot_results():
    Path("./plots").mkdir(exist_ok=True)
    topn_limiter = [1, 5, 10]
    report = pkl.load(open("evaluation_report.pkl", "rb"))
    plot_tables = {}
    for data_set_name in report.keys():
        for prune_technique in report[data_set_name].keys():
            for split_technique in report[data_set_name][prune_technique].keys():
                for recommender in report[data_set_name][prune_technique][split_technique].keys():
                    plot_rows = []
                    for recommender_seed in report[data_set_name][prune_technique][split_technique][recommender].keys():
                        for shuffle_seed in report[data_set_name][prune_technique][split_technique][recommender][
                            recommender_seed].keys():
                            aggregated_results = {}
                            for test_fold in \
                                    report[data_set_name][prune_technique][split_technique][recommender][
                                        recommender_seed][
                                        shuffle_seed].keys():
                                for run_batch in \
                                        report[data_set_name][prune_technique][split_technique][recommender][
                                            recommender_seed][
                                            shuffle_seed][test_fold].keys():
                                    for topn_score in \
                                            report[data_set_name][prune_technique][split_technique][recommender][
                                                recommender_seed][shuffle_seed][test_fold][run_batch].keys():
                                        if topn_score in topn_limiter:
                                            if test_fold not in aggregated_results:
                                                aggregated_results[test_fold] = {}
                                            if topn_score not in aggregated_results[test_fold]:
                                                aggregated_results[test_fold][topn_score] = []
                                            aggregated_results[test_fold][topn_score].append(
                                                report[data_set_name][prune_technique][split_technique][recommender][
                                                    recommender_seed][shuffle_seed][test_fold][run_batch][topn_score])
                            for test_fold in aggregated_results.keys():
                                for topn_score in aggregated_results[test_fold].keys():
                                    aggregated_results[test_fold][topn_score] = pd.DataFrame(
                                        aggregated_results[test_fold][topn_score])
                            topn_scores = list(aggregated_results[0].keys())
                            # go through entries and create plotting table
                            for topn_score in topn_scores:
                                holdout_precision = aggregated_results[0][topn_score]["precision"].mean()
                                holdout_ndcg = aggregated_results[0][topn_score]["ndcg"].mean()
                                cv_precision = np.array(
                                    [aggregated_results[fold][topn_score]["precision"].mean() for fold in
                                     aggregated_results.keys()]).mean()
                                cv_ndcg = np.array([aggregated_results[fold][topn_score]["ndcg"].mean() for fold in
                                                    aggregated_results.keys()]).mean()
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
                    # normalize metric values
                    for recommender_seed in plot_table["recommender_seed"].unique():
                        for k in plot_table["topn_score"].unique():
                            for validation in plot_table["validation_type"].unique():
                                for metric in plot_table["metric"].unique():
                                    metric_vals = \
                                        plot_table[(plot_table["recommender_seed"] == recommender_seed) &
                                                   (plot_table["topn_score"] == k) &
                                                   (plot_table["validation_type"] == validation) &
                                                   (plot_table["metric"] == metric)]["metric_value"]
                                    min_val = metric_vals.min()
                                    max_val = metric_vals.max()
                                    mean_val = metric_vals.mean()
                                    plot_table.loc[
                                        (plot_table["recommender_seed"] == recommender_seed) &
                                        (plot_table["topn_score"] == k) &
                                        (plot_table["validation_type"] == validation) &
                                        (plot_table["metric"] == metric),
                                        "metric_value_relative_min"] = ((metric_vals / min_val) - 1) * 100
                                    plot_table.loc[
                                        (plot_table["recommender_seed"] == recommender_seed) &
                                        (plot_table["topn_score"] == k) &
                                        (plot_table["validation_type"] == validation) &
                                        (plot_table["metric"] == metric),
                                        "metric_value_relative_max"] = metric_vals / max_val
                                    plot_table.loc[
                                        (plot_table["recommender_seed"] == recommender_seed) &
                                        (plot_table["topn_score"] == k) &
                                        (plot_table["validation_type"] == validation) &
                                        (plot_table["metric"] == metric),
                                        "metric_value_relative_mean"] = (metric_vals / mean_val) * 100
                                    plot_table.loc[
                                        (plot_table["recommender_seed"] == recommender_seed) &
                                        (plot_table["topn_score"] == k) &
                                        (plot_table["validation_type"] == validation) &
                                        (plot_table["metric"] == metric),
                                        "metric_value_absolute_mean"] = abs(100 - ((metric_vals / mean_val) * 100))
                    plot_table.rename(columns={'recommender_seed': "Training Seed", 'shuffle_seed': "Data Shuffle Seed",
                                               'topn_score': "k", 'validation_type': "Validation",
                                               'metric': "Metric", 'metric_value': "Metric Value",
                                               "metric_value_relative_min": "Relative Metric Value (Min)",
                                               "metric_value_relative_max": "Relative Metric Value (Max)",
                                               "metric_value_relative_mean": "Relative Metric Value (Mean)",
                                               "metric_value_absolute_mean": "Absolute Metric Value (Mean)"},
                                      inplace=True)
                    if data_set_name not in plot_tables:
                        plot_tables[data_set_name] = {}
                    if prune_technique not in plot_tables[data_set_name]:
                        plot_tables[data_set_name][prune_technique] = {}
                    if split_technique not in plot_tables[data_set_name][prune_technique]:
                        plot_tables[data_set_name][prune_technique][split_technique] = {}
                    if recommender not in plot_tables[data_set_name][prune_technique][split_technique]:
                        plot_tables[data_set_name][prune_technique][split_technique][recommender] = plot_table

                    '''
                    # print stats of metric values
                    stat_table_rows = []
                    for recommender_seed in plot_table["Training Seed"].unique():
                        for k in plot_table["k"].unique():
                            for validation in plot_table["Validation"].unique():
                                for metric in plot_table["Metric"].unique():
                                    relevant_data = plot_table[
                                        (plot_table["Training Seed"] == recommender_seed) & (plot_table["k"] == k) & (
                                                plot_table["Validation"] == validation) & (
                                                plot_table["Metric"] == metric)]
                                    stat_table_rows.append({"k": k,
                                                            "Validation": validation,
                                                            "Metric": metric,
                                                            "Mean": relevant_data["Metric Value"].mean(),
                                                            "Std": relevant_data["Metric Value"].std(),
                                                            "Var": relevant_data["Metric Value"].var(),
                                                            "Range": relevant_data["Metric Value"].max() -
                                                                     relevant_data["Metric Value"].min()})
                    stat_table = pd.DataFrame(stat_table_rows)
                    latex_table = stat_table.to_latex(index=False)
                    '''

                    '''
                    # plot absolute results
                    sns.set(font_scale=.5)
                    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
                    cat = sns.catplot(
                        data=plot_table,
                        x="Metric Value", y="Validation", hue="Data Shuffle Seed", row="k", col="Metric",
                        palette="colorblind", height=1, aspect=3, s=10)
                    plt.subplots_adjust(top=0.85)
                    cat.fig.suptitle(f"{data_set_name} - {recommender}")
                    plt.savefig(f'./plots/_absolute_{data_set_name}-{recommender}.pdf', bbox_inches="tight")
                    # plot relative results
                    cat = sns.catplot(
                        data=plot_table,
                        x="Relative Metric Value (Mean)", y="Validation", hue="Data Shuffle Seed", row="k", col="Metric",
                        palette="colorblind", height=1, aspect=3, s=10)
                    plt.subplots_adjust(top=0.85)
                    cat.fig.suptitle(f"{data_set_name} - {recommender}")
                    plt.savefig(f'./plots/_relative_mean_{data_set_name}-{recommender}.pdf', bbox_inches="tight")
                    cat = sns.catplot(
                        data=plot_table,
                        x="Relative Metric Value (Max)", y="Validation", hue="Data Shuffle Seed", row="k",
                        col="Metric",
                        palette="colorblind", height=1, aspect=3, s=10)
                    plt.subplots_adjust(top=0.85)
                    cat.fig.suptitle(f"{data_set_name} - {recommender}")
                    plt.savefig(f'./plots/_relative_max_{data_set_name}-{recommender}.pdf', bbox_inches="tight")
                    '''

    aggregated_results = {}
    for data_set_name in plot_tables.keys():
        for prune_technique in plot_tables[data_set_name].keys():
            if prune_technique not in aggregated_results:
                aggregated_results[prune_technique] = {}
            for split_technique in plot_tables[data_set_name][prune_technique].keys():
                if split_technique not in aggregated_results[prune_technique]:
                    aggregated_results[prune_technique][split_technique] = {}
                for recommender in plot_tables[data_set_name][prune_technique][split_technique].keys():
                    if recommender not in aggregated_results[prune_technique][split_technique]:
                        aggregated_results[prune_technique][split_technique][recommender] = pd.DataFrame()
                    aggregated_results[prune_technique][split_technique][recommender] = pd.concat(
                        [aggregated_results[prune_technique][split_technique][recommender],
                         plot_tables[data_set_name][prune_technique][split_technique][recommender]],
                        ignore_index=True)

    for prune_technique in aggregated_results.keys():
        for split_technique in aggregated_results[prune_technique].keys():
            for recommender in aggregated_results[prune_technique][split_technique].keys():
                this_table = aggregated_results[prune_technique][split_technique][recommender]
                holdout_values = this_table[this_table["Validation"] == "holdout"][
                    "Absolute Metric Value (Mean)"].values
                cross_validation_values = this_table[this_table["Validation"] == "cross-validation"][
                    "Absolute Metric Value (Mean)"].values
                statistical_test = wilcoxon(holdout_values, cross_validation_values)
                print(f"{prune_technique} - {split_technique} - {recommender} wilcoxon test "
                      f"holdout versus cross validation\n"
                      f"statistic:{statistical_test.statistic} - pvalue:{statistical_test.pvalue}")

    for prune_technique in aggregated_results.keys():
        for split_technique in aggregated_results[prune_technique].keys():
            for recommender in aggregated_results[prune_technique][split_technique].keys():
                relevant_data = aggregated_results[prune_technique][split_technique][recommender]
                relevant_data.loc[relevant_data["Validation"] == "cross-validation", "Validation"] = "CV"
                relevant_data.loc[relevant_data["Validation"] == "holdout", "Validation"] = "HO"
                relevant_data.loc[relevant_data["Metric"] == "Precision", "Metric"] = "Precision@k"
                relevant_data.loc[relevant_data["Metric"] == "nDCG", "Metric"] = "nDCG@k"

                recommender_title = recommender
                if recommender == "implicit-mf":
                    recommender_title = "Implicit Matrix Factorization with Alternating Least Squares (ALS)"
                elif recommender == "item-knn":
                    recommender_title = "Item-based k-Nearest Neighbors (ItemKNN)"
                elif recommender == "popularity":
                    recommender_title = "Popularity Recommender (Pop)"

                print(recommender_title)

                # plot relative results
                sns.set(font_scale=.5)
                sns.set_style("ticks")
                sns.set_style("whitegrid",
                              {"grid.color": ".8", "grid.linestyle": "--", "grid.lineWidth": ".3", "xtick.bottom": True,
                               'xtick.color': 'grey'})

                cat = sns.catplot(
                    data=relevant_data,
                    x="Relative Metric Value (Min)", y="Validation", row="k", col="Metric",
                    palette="colorblind", height=1, aspect=3, kind="box", fliersize=0.8, linewidth=0.5, whis=1.5)
                cat.set_axis_labels('', '')
                cat.fig.supylabel('Validation Method')
                cat.fig.supxlabel('Accuracy Depending on Data Split Random Seed - Relative to Minimum Accuracy in %')
                plt.subplots_adjust(top=0.88)
                cat.fig.suptitle(f"{recommender_title}")
                max_value = relevant_data.loc[relevant_data["Validation"] ==
                                              "HO", "Relative Metric Value (Min)"].max()
                print(f"Max value HO: {max_value}")
                max_value_cv = relevant_data.loc[relevant_data["Validation"] ==
                                                 "CV", "Relative Metric Value (Min)"].max()
                print(f"Max value CV: {max_value_cv}")
                cat.set(xlim=(-1, max_value + 1))
                for ax in cat.axes.flat:
                    cutoff, metric = ax.get_title().split(" |")
                    ax.set_title(f"{metric.split(' = ')[1][:-2]}@{cutoff.split('= ')[1]}")
                    ax.axhline(0.5, color='grey', linestyle='-', linewidth=0.6)
                cat.fig.subplots_adjust(wspace=0.1, hspace=0.5)
                plt.savefig(f'./plots/_agg-min-{recommender}.pdf', bbox_inches="tight")

                cat = sns.catplot(
                    data=relevant_data,
                    x="Relative Metric Value (Max)", y="Validation", row="k", col="Metric",
                    palette="colorblind", height=1, aspect=3, kind="box", fliersize=0.8, linewidth=0.5, whis=1.5)
                cat.set_axis_labels('', '')
                cat.fig.supylabel('Validation Method')
                cat.fig.supxlabel('Accuracy Depending on Data Split Random Seed - Relative to Maximum Accuracy in %')
                plt.subplots_adjust(top=0.88)
                cat.fig.suptitle(f"{recommender_title}")
                min_value = relevant_data.loc[relevant_data["Validation"] ==
                                              "HO", "Relative Metric Value (Max)"].min()
                print(f"Min value HO: {min_value}")
                min_value_cv = relevant_data.loc[relevant_data["Validation"] ==
                                                 "CV", "Relative Metric Value (Max)"].min()
                print(f"Min value CV: {min_value_cv}")
                cat.set(xlim=(min_value - 0.01, 1.01))
                for ax in cat.axes.flat:
                    cutoff, metric = ax.get_title().split(" |")
                    ax.set_title(f"{metric.split(' = ')[1][:-2]}@{cutoff.split('= ')[1]}")
                    ax.axhline(0.5, color='grey', linestyle='-', linewidth=0.6)
                cat.fig.subplots_adjust(wspace=0.1, hspace=0.5)
                plt.savefig(f'./plots/_agg-max-{recommender}.pdf', bbox_inches="tight")

                cat = sns.catplot(
                    data=relevant_data,
                    x="Relative Metric Value (Mean)", y="Validation", row="k", col="Metric",
                    palette="colorblind", height=1, aspect=3, kind="box", fliersize=0.8, linewidth=0.5, whis=1.5)
                cat.set_axis_labels('', '')
                cat.fig.supylabel('Validation Method')
                cat.fig.supxlabel('Accuracy Depending on Data Split Random Seed - Relative to Mean Accuracy in %')
                plt.subplots_adjust(top=0.88)
                cat.fig.suptitle(f"{recommender_title}")
                range_to_max = relevant_data.loc[relevant_data["Validation"] ==
                                                 "HO", "Relative Metric Value (Mean)"].max() - 100
                print(f"Range to max HO: {range_to_max}")
                range_to_max_cv = relevant_data.loc[relevant_data["Validation"] ==
                                                    "CV", "Relative Metric Value (Mean)"].max() - 100
                print(f"Range to max CV: {range_to_max_cv}")
                range_to_min = 100 - relevant_data.loc[relevant_data["Validation"] ==
                                                       "HO", "Relative Metric Value (Mean)"].min()
                print(f"Range to min HO: {range_to_min}")
                range_to_min_cv = 100 - relevant_data.loc[relevant_data["Validation"] ==
                                                          "CV", "Relative Metric Value (Mean)"].min()
                print(f"Range to min CV: {range_to_min_cv}")
                maximum_range = max(range_to_max, range_to_min) * 1.1
                cat.set(xlim=(100 - maximum_range, 100 + maximum_range))
                for ax in cat.axes.flat:
                    cutoff, metric = ax.get_title().split(" |")
                    ax.set_title(f"{metric.split(' = ')[1][:-2]}@{cutoff.split('= ')[1]}")
                    ax.axhline(0.5, color='grey', linestyle='-', linewidth=0.6)
                cat.fig.subplots_adjust(wspace=0.1, hspace=0.5)
                plt.savefig(f'./plots/_agg-mean-{recommender}.pdf', bbox_inches="tight")

    return
