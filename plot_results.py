import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle as pkl


def plot_results():
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
                                    max_val = metric_vals.max()
                                    mean_val = metric_vals.mean()
                                    plot_table.loc[
                                        (plot_table["recommender_seed"] == recommender_seed) &
                                        (plot_table["topn_score"] == k) &
                                        (plot_table["validation_type"] == validation) &
                                        (plot_table[
                                             "metric"] == metric), "metric_value_relative_max"] = metric_vals / max_val
                                    plot_table.loc[
                                        (plot_table["recommender_seed"] == recommender_seed) &
                                        (plot_table["topn_score"] == k) &
                                        (plot_table["validation_type"] == validation) &
                                        (plot_table[
                                             "metric"] == metric), "metric_value_relative_mean"] = metric_vals / mean_val
                    plot_table.rename(columns={'recommender_seed': "Training Seed", 'shuffle_seed': "Data Shuffle Seed",
                                               'topn_score': "k", 'validation_type': "Validation",
                                               'metric': "Metric", 'metric_value': "Metric Value",
                                               "metric_value_relative_max": "Relative Metric Value (Max)",
                                               "metric_value_relative_mean": "Relative Metric Value (Mean)"},
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

                    # plot absolute results
                    sns.set(font_scale=.5)
                    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
                    cat = sns.catplot(
                        data=plot_table,
                        x="Metric Value", y="Validation", hue="Data Shuffle Seed", row="k", col="Metric",
                        palette="colorblind", height=1, aspect=3, s=10)
                    plt.subplots_adjust(top=0.85)
                    cat.fig.suptitle(f"{data_set_name} - {recommender}")
                    plt.savefig(f'_absolute_{data_set_name}-{recommender}.pdf', bbox_inches="tight")
                    # plot relative results
                    cat = sns.catplot(
                        data=plot_table,
                        x="Relative Metric Value", y="Validation", hue="Data Shuffle Seed", row="k", col="Metric",
                        palette="colorblind", height=1, aspect=3, s=10)
                    plt.subplots_adjust(top=0.85)
                    cat.fig.suptitle(f"{data_set_name} - {recommender}")
                    plt.savefig(f'_relative_{data_set_name}-{recommender}.pdf', bbox_inches="tight")

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
                relevant_data = aggregated_results[prune_technique][split_technique][recommender]
                '''
                for k in relevant_data["k"].unique():
                    for metric in relevant_data["Metric"].unique():
                        # seaborn box plot
                        sns.set(font_scale=.5)
                        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
                        relevant_data_c = relevant_data[(relevant_data["k"] == k) & (relevant_data["Metric"] == metric)].drop(columns=["Data Shuffle Seed"])
                        box = sns.boxplot(data=relevant_data_c,
                                          x="Relative Metric Value", y="Validation")
                        plt.show()
                        print()
                '''
                # plot relative results
                sns.set(font_scale=.5)
                sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
                cat = sns.catplot(
                    data=relevant_data,
                    x="Relative Metric Value (Mean)", y="Validation", row="k", col="Metric",
                    palette="colorblind", height=1, aspect=3, kind="box", fliersize=0.8, linewidth=0.5, whis=1.5)
                plt.subplots_adjust(top=0.85)
                cat.fig.suptitle(f"{recommender} aggregated")
                range_to_max = relevant_data["Relative Metric Value (Mean)"].max() - 1
                range_to_min = 1 - relevant_data["Relative Metric Value (Mean)"].min()
                maximum_range = max(range_to_max, range_to_min)*1.1
                cat.set(xlim=(1 - maximum_range, 1 + maximum_range))
                plt.savefig(f'_agg-{recommender}.pdf', bbox_inches="tight")

    print()
