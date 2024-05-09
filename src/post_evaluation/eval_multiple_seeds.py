
import pandas as pd
import numpy as np
import os
import os.path as osp
from collections import defaultdict
import matplotlib.pyplot as plt
# read csv files of multiple seeds
from post_evaluation.plotting import plot_recall_per_object_type, plot_ap, plot_prec_recall_data, bar_plot

FOLDER = "../scobots_spaceandmoc_detectors"
# FOLDER = "../output/logs"

def get_dataframes_per_game(filename, csv_separator, index_col):
    folder = FOLDER
    # get subfolders
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir() ]
    subfolders.sort(key=lambda x: int(x.split("seed")[-1]))

    # get subfolders for each game
    subfolders_per_game = defaultdict(list)
    for subfolder in subfolders:
        if "seed" not in subfolder:
            continue
        game = subfolder.split("/")[-1].split("_")[0]
        subfolders_per_game[game].append(subfolder)

    dataframes_per_game = defaultdict(list)
    for game, subfolders in subfolders_per_game.items():
        for subfolder in subfolders:
            path = osp.join(subfolder, filename)
            dataframes_per_game[game].append(pd.read_csv(path, sep=csv_separator, index_col=index_col))
    return dataframes_per_game


def get_dataframes_per_game_best_seed(filename, csv_separator, index_col, intermediate_folder=None):
    folder = FOLDER
    # get subfolders
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir() ]
    games = [folder.split("/")[-1] for folder in subfolders]


    dataframes_per_game = defaultdict(list)
    for game in games:
        if intermediate_folder is not None:
            path = osp.join(folder, game, intermediate_folder, filename)
        else:
            path = osp.join(folder, game, filename)
        dataframes_per_game[game].append(pd.read_csv(path, sep=csv_separator, index_col=index_col))
    return dataframes_per_game


def add_fscore_column(df, styles=['relevant', 'all']):
    for style in styles:
        if not df[f'{style}_precision'].empty:
            df[f'{style}_f_score'] = 2 * df[f'{style}_precision'] * df[
                f'{style}_recall'] / (df[f'{style}_precision'] +
                                      df[f'{style}_recall'] + 1e-8)
        else:
            df[f'{style}_f_score'] = np.nan
    return df

def add_ap_avg_column(df, styles=['relevant', 'all',]):
    for style in styles:
        aps = df.filter(regex=f'{style}_ap')
        df[f'{style}_ap_avg'] = aps.mean(axis=1)
    return df

def add_contrived_columns(dataframes_per_game):
    # add f_score and ap_avg columns
    for game, dataframes in dataframes_per_game.items():
        dataframes = [add_fscore_column(df) for df in dataframes]
        dataframes = [add_ap_avg_column(df) for df in dataframes]
        dataframes_per_game[game] = dataframes
    return dataframes_per_game


def compute_mean_and_std(dataframe_per_game):
    # columns_mean_std are all columns in dataframe_per_game

    final_dataframe_per_game = {}
    for game, dataframe in dataframe_per_game.items():
        dict_mean_std = {}
        for column in dataframe.columns:
            values = dataframe[column].values
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)
            dict_mean_std[f"{column}_mean"] = mean
            dict_mean_std[f"{column}_std"] = std
        final_dataframe_per_game[game] = pd.DataFrame(dict_mean_std, index=[0])
    return final_dataframe_per_game



def merge_dataframes_per_game(dataframes_per_game):
    # merge dataframes for each game
    merged_dataframes_per_game = {}
    for game, dataframes in dataframes_per_game.items():
        dataframes = [df.iloc[[-1]] for df in dataframes]
        merged_dataframes_per_game[game] = pd.concat(dataframes)
    return merged_dataframes_per_game

def preprocess_eval_classifier_dataframes(dataframes_per_game):
    modified_dataframes_per_game = {}
    for game, dataframes in dataframes_per_game.items():
        ## transpose dataframes
        #dataframes = [df.T for df in dataframes]
        ## select row named "macro avg
        #dataframes = [df.loc[["macro avg"]] for df in dataframes] #TODO check which avg to use; probably rather micro avg but this is not available currently
        dataframes = [pd.DataFrame({"relevant_accuracy": [df.loc["precision", "accuracy"]]}) for df in dataframes]
        modified_dataframes_per_game[game] = dataframes
    return modified_dataframes_per_game



baseline_results = {
    "pong": {
        "relevant_f_score_mean": 92.8,
        "relevant_f_score_std":  1.3,
        "relevant_adjusted_mutual_info_score_mean": 0.0,
        "relevant_adjusted_mutual_info_score_std": 0.0,
    },
    "boxing": {
        "relevant_f_score_mean": 92.1,
        "relevant_f_score_std": 0.7,
        "relevant_adjusted_mutual_info_score_mean": 0.0,
        "relevant_adjusted_mutual_info_score_std": 0.0,
    },
    "skiing": {
        "relevant_f_score_mean": np.nan,
        "relevant_f_score_std": np.nan,
        "relevant_adjusted_mutual_info_score_mean": np.nan,
        "relevant_adjusted_mutual_info_score_std": np.nan,
    },
}

# taken from "Relevant CLuster Classifier Accuracy" table (Table 19) from MOC paper
classifier_baseline_results = {
    "pong": {
        "relevant_accuracy_mean": 98.7,
        "relevant_accuracy_std": 0.4,   
    },
    "boxing": {
        "relevant_accuracy_mean": 96.4,
        "relevant_accuracy_std": 0.5,
    },
    "skiing": {
        "relevant_accuracy_mean": np.nan,
        "relevant_accuracy_std": np.nan,
    },
}

if __name__ == "__main__":
    category = "relevant"
    # dataframes_per_game = get_dataframes_per_game("test_metrics.csv", csv_separator=";", index_col=None)
    # dataframes_per_game = add_contrived_columns(dataframes_per_game)
    # merged_dataframes_per_game = merge_dataframes_per_game(dataframes_per_game)
    # final_dataframe_per_game = compute_mean_and_std(merged_dataframes_per_game)
    # bar_plot(final_dataframe_per_game, baseline_results, metric="relevant_f_score", title="Localization", ylabel="F-Score (%)")
    # plot_prec_recall_data(final_dataframe_per_game)
    # plot_ap(final_dataframe_per_game)
    # plot_recall_per_object_type(final_dataframe_per_game)
    # bar_plot(final_dataframe_per_game, baseline_results, metric="relevant_adjusted_mutual_info_score", title="Mutual Information", ylabel="Mutual Information (%)")


    # dataframes_per_game = get_dataframes_per_game_best_seed("test_metrics.csv", csv_separator=";", index_col=None)
    # dataframes_per_game = add_contrived_columns(dataframes_per_game)
    # merged_dataframes_per_game = merge_dataframes_per_game(dataframes_per_game)
    # final_dataframe_per_game = compute_mean_and_std(merged_dataframes_per_game)
    # plot_prec_recall_data(final_dataframe_per_game)
    # plot_recall_per_object_type(final_dataframe_per_game)


    # eval classifier
    # dataframes_per_game = get_dataframes_per_game("eval_classifier.csv", csv_separator=",", index_col=0)
    dataframes_per_game = get_dataframes_per_game_best_seed("eval_classifier.csv", csv_separator=",", index_col=0, intermediate_folder="classifier")
    dataframes_per_game = preprocess_eval_classifier_dataframes(dataframes_per_game)
    merged_dataframes_per_game = merge_dataframes_per_game(dataframes_per_game)
    final_dataframe_per_game = compute_mean_and_std(merged_dataframes_per_game)
    bar_plot(final_dataframe_per_game, classifier_baseline_results, metric="relevant_accuracy", title="Classifier Accuracy", ylabel="Accuracy (%)")

    # eval model and classifier
    # dataframes_per_game = get_dataframes_per_game("eval_model_and_classifier.csv", csv_separator=",", index_col=0)
    dataframes_per_game = get_dataframes_per_game_best_seed("eval_model_and_classifier.csv", csv_separator=",", index_col=0, intermediate_folder="space_weights")
    merged_dataframes_per_game = merge_dataframes_per_game(dataframes_per_game)
    final_dataframe_per_game = compute_mean_and_std(merged_dataframes_per_game)
    bar_plot(final_dataframe_per_game, metric="relevant_f1_score", title="Detection", ylabel="F-Score (%)")
    bar_plot(final_dataframe_per_game, metric="relevant_precision", title="Detection", ylabel="Precision (%)")
    bar_plot(final_dataframe_per_game, metric="relevant_recall", title="Detection", ylabel="Recall (%)")
    print("done")




