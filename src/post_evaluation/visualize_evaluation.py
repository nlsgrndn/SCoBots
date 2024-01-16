import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import matplotlib.colors as mcolors
import re
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import argparse
from termcolor import colored
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--save',
                    '-s',
                    default=False,
                    action="store_true",
                    help='Save the image(s) instead of showing them')
parser.add_argument(
    '--splitted',
    default=False,
    action="store_true",
    help='Save individual image(s) instead of one and generate associated tex')
parser.add_argument('--final-test',
                    default=False,
                    action="store_true",
                    help='Use final test evaluation')
# parser.add_argument('--num-frame-stack', type=int, default=1,
#                     help='Number of frames to stack for a state')

args = parser.parse_args()

RESULT_TEX = os.path.join("..", "results_img", "result.tex")
sns.set_theme()
if args.final_test:
    data_path = "../final_test_results"
    result_path = '../final_test_results_img'
else:
    data_path = "../results"
    result_path = '../results_img'

def add_contrived_columns(df, styles=['relevant', 'all', 'moving']):
    for style in styles:
        aps = df.filter(regex=f'{style}_ap')
        df[f'{style}_ap_avg'] = aps.mean(axis=1)

        if not df[f'{style}_precision'].empty:
            df[f'{style}_f_score'] = 2 * df[f'{style}_precision'] * df[
                f'{style}_recall'] / (df[f'{style}_precision'] +
                                      df[f'{style}_recall'] + 1e-8)
        else:
            df[f'{style}_f_score'] = np.nan
    return df

def draw_precision_recall_curve(precision, recall, save_path,):
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(save_path)
    plt.close()

def visualize_APs(aps, iou_thresholds, save_path):
    plt.plot(iou_thresholds, aps)
    plt.xlabel('IOU Thresholds')
    plt.ylabel('AP')
    plt.title('APs')
    plt.savefig(save_path)
    plt.close()


def plot_ap(df):
    fig, ax = plt.subplots()
    categories = ['relevant', 'all', 'moving']
    
    for category in categories:
        ap = df.filter(regex=f'{category}_ap')
        #additionally remove column f'{category}_ap_avg'
        ap = ap.drop(columns=[f'{category}_ap_avg'])
        #rename columns to only keep last element when split by '_'
        ap.columns = ap.columns.str.split('_').str[-1]
        # first row in column are the ious, secnd row are the aps
        ap = ap.T
        #interpret index as number
        ap.index = ap.index.astype(float)
        ap.plot(ax=ax, marker='o', label=category)
        #plot the average ap as a horizontal line
        ax.axhline(df[f'{category}_ap_avg'].mean(),
                   linestyle='--',
                   label=f'{category}_ap_avg', # use same color as for category
                   color=ax.lines[-1].get_color())

    ax.set_xlabel('IOU')
    ax.set_ylabel('AP')
    ax.set_title('APs')
    plt.savefig(os.path.join(result_path, "img", f'APs.png'))


def collect_prec_recall_data(df):
    
    categories = ['relevant', 'all', 'moving']

    for category in categories:
        rec = df.filter(regex=f'{category}_recall_')
        prec = df.filter(regex=f'{category}_precision_')
        rec.columns = rec.columns.str.split('_').str[-1]
        prec.columns = prec.columns.str.split('_').str[-1]
        thresholds = rec.columns.astype(float)
        recall_values = rec.iloc[0]
        precision_values = prec.iloc[0]
        draw_precision_recall_curve(recall_values, precision_values, thresholds, os.path.join(result_path, "img", f'precision_recall_curve_{category}.png'), category)
        
def draw_precision_recall_curve(recalls, precisions, thresholds, save_path, category):
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions)
    #annotate the thresholds
    for i, threshold in enumerate(thresholds):
        ax.annotate(np.round(threshold, decimals=2), (recalls[i], precisions[i]))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve for {category}')
    plt.savefig(save_path)

def main():
    if os.path.exists(RESULT_TEX):
        os.remove(RESULT_TEX)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(result_path, "img"), exist_ok=True)
    files = os.listdir(data_path)
    files = [f for f in files if f.endswith(".csv")] # collect files from multiple seeds or games
    #file = files[0] # only one file for now
    f_scores = {}
    data_subset_modes = ["all", "relevant"]
    for file in files:
        print(f"Processing {file}")
        df = pd.read_csv(os.path.join(data_path, file), sep=";")
        df = add_contrived_columns(df, data_subset_modes)
        # drop all but the last row 
        df = df.iloc[[-1]]
        for category in data_subset_modes:
            print(f"{category}_f_score:", df[f'{category}_f_score'].values[0])

            f_scores[file.split("_")[0]] = df[f'{category}_f_score'].values[0]

            try:
                import ipdb; ipdb.set_trace()
                recall_values_per_object_type = [df[f'{category}_recall_label_{label}'].values[0] for label in range(6)]
                plot_recall_per_object_type(category, recall_values_per_object_type)
            except:
                print("No recall values per object type available")


            try:
                # print few shot accuracy
                for i in [1, 4, 16, 64]:
                    print(f"{category}_few_shot_accuracy_with_{i}:", df[f'{category}_few_shot_accuracy_with_{i}'].values[0])
                print(f"{category}_few_shot_accuracy_cluster_nn:", df[f'{category}_few_shot_accuracy_cluster_nn'].values[0])
                # print mutual information
                print(f"{category}_adjusted_rand_score", df[f'{category}_adjusted_rand_score'].values[0])
                print(f"{category}_adjusted_mutual_info_score", df[f'{category}_adjusted_mutual_info_score'].values[0])
            except:
                print("No few shot accuracy or mutual information available")

        #plot_ap(df)
        #collect_prec_recall_data(df)
    plot_f_scores(f_scores)

def plot_f_scores(f_scores):
    # Extracting the game names and their corresponding f-scores
    games = list(f_scores.keys())
    scores = list(f_scores.values())

    # Using the object-oriented approach in Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(games, scores, color='skyblue')

    # Setting labels and title
    ax.set_xlabel('Games')
    ax.set_ylabel('F-Scores')
    ax.set_title('F-Scores of Various Games')
    ax.set_ylim(0, 1)

    # save the figure
    plt.savefig(os.path.join(result_path, "img", f'f_scores.png'))

def plot_recall_per_object_type(data_subset_mode, recall_values_per_object_type):
    # Extracting the object types and their corresponding recall values
    
    #filter out nan values
    #recall_values_per_object_type = [recall_value for recall_value in recall_values_per_object_type if not np.isnan(recall_value)]

    object_types = [str(i) for i in range(len(recall_values_per_object_type))]
    recall_values = recall_values_per_object_type

    # Using the object-oriented approach in Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(object_types, recall_values, color='skyblue')

    # Setting labels and title
    ax.set_xlabel('Object Types')
    ax.set_ylabel('Recall Values')
    ax.set_title(f'Recall Values of {data_subset_mode} Object Types')
    ax.set_ylim(0, 1)

    # save the figure
    plt.savefig(os.path.join(result_path, "img", f'recall_values_{data_subset_mode}.png'))

if __name__ == "__main__":
    main()