import pandas as pd
import numpy as np
import os
import os.path as osp
from collections import defaultdict
import matplotlib.pyplot as plt


result_path = '../results_img'

def plot_recall_per_object_type(final_dataframe_per_game):
    # Extracting the object types and their corresponding recall values
    
    #filter out nan values
    #recall_values_per_object_type = [recall_value for recall_value in recall_values_per_object_type if not np.isnan(recall_value)]
    category = "relevant"

    for game, df in final_dataframe_per_game.items():
        # filter df for relevant object types
        df = df.filter(regex=f'{category}_recall_label_\d+_mean')
        recall_values_per_object_type = df.values[0]
        #recall_values_per_object_type = [df[f'{category}_recall_label_{label}_mean'].values[0] for label in range(6)]

        object_types = [str(i) for i in range(len(recall_values_per_object_type))]
        recall_values = recall_values_per_object_type

        # Using the object-oriented approach in Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(object_types, recall_values, color='skyblue')

        # Setting labels and title
        ax.set_xlabel('Object Types')
        ax.set_ylabel('Recall Values')
        ax.set_title(f'Recall Values of {category} Object Types')
        ax.set_ylim(0, 1)

        # save the figure
        plt.savefig(f"recall_per_object_type_{game}.png")

def draw_precision_recall_curve(recalls, precisions, thresholds, save_path, game):
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions)
    #annotate the thresholds
    for i, threshold in enumerate(thresholds):
        ax.annotate(np.round(threshold, decimals=2), (recalls[i], precisions[i]))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve for {game}')
    plt.savefig(save_path)

def plot_ap(final_dataframe_per_game):
    fig, ax = plt.subplots()
    for game, df in final_dataframe_per_game.items():
        category ="relevant"
        ap = df.filter(regex=f'{category}_ap_\d+\.\d+_mean')
        ap.columns = ap.columns.str.split('_').str[-2] #-2 because of the _mean suffix
        ap = ap.T # first row in column are the ious, secnd row are the aps
        ap.index = ap.index.astype(float) #interpret index as number
        
        ap.columns = [game] # set name for plot legend
        ap.plot(ax=ax, marker='o')
        
        #plot the average ap as a horizontal line
        ax.axhline(df[f'{category}_ap_avg_mean'].mean(),
                   linestyle='--',
                   color=ax.lines[-1].get_color()) # use same color for the horizontal line as for the plot line

    ax.set_xlabel('IOU')
    ax.set_ylabel('AP')
    ax.set_title('APs')
    plt.savefig("aps.png")


def bar_plot(dataframe_per_game, baseline_results = None, metric="relevant_f_score", title="Localization", ylabel="F-Score (%)"):
    if baseline_results:
        baseline_dataframe_per_game = {k: pd.DataFrame(v, index=[0]) for k, v in baseline_results.items()}
    fig, ax = plt.subplots()

    games = dataframe_per_game.keys()

    if baseline_results:
        baseline_values = []
        baseline_stds = []
    our_values = []
    our_stds = []
    labels = []

    x = np.arange(len(games))  # the label locations
    width = 0.35  # the width of the bars

    for game in dataframe_per_game.keys():
        if baseline_results:
            baseline_value = baseline_dataframe_per_game[game][f"{metric}_mean"].values[0]
            baseline_std = baseline_dataframe_per_game[game][f"{metric}_std"].values[0]
            baseline_values.append(baseline_value)
            baseline_stds.append(baseline_std)
        our_value = dataframe_per_game[game][f"{metric}_mean"].values[0]
        our_std = dataframe_per_game[game][f"{metric}_std"].values[0]

        our_values.append(our_value)
        our_stds.append(our_std)
        labels.append(game)

    # transform to percentage
    our_values = np.array(our_values) * 100
    our_stds = np.array(our_stds) * 100
    # not done for baseline because it is already in percentage
    if baseline_results:
        rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='orange', yerr=baseline_stds, capsize=5)
        rects2 = ax.bar(x + width/2, our_values, width, label='SPACE + MOC', color='blue', yerr=our_stds, capsize=5)
    else:
        rects2 = ax.bar(x, our_values, width, label='SPACE + MOC', color='blue', yerr=our_stds, capsize=5)

    # figure formatting
    ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.2,)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(games)
    ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    baseline_str = "_baseline" if baseline_results else ""
    plt.savefig(f"bar_plot_{metric}{baseline_str}.png")


def plot_prec_recall_data(final_dataframe_per_game):
    category = "relevant"
    for game, df in final_dataframe_per_game.items():  
        rec = df.filter(regex=f'{category}_recall_\d+\.\d+_mean')
        prec = df.filter(regex=f'{category}_precision_\d+\.\d+_mean')
        rec.columns = rec.columns.str.split('_').str[-2] #-2 because of the _mean and _std suffix
        prec.columns = prec.columns.str.split('_').str[-2]
        thresholds = rec.columns.astype(float)
        recall_values = rec.iloc[0]
        precision_values = prec.iloc[0]
        draw_precision_recall_curve(recall_values, precision_values, thresholds, f'precision_recall_curve_{game}.png', game)