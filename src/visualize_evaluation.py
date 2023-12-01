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

def add_contrived_columns(df):
    for style in ['relevant', 'all', 'moving']:
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
        draw_precision_recall_curve(recall_values, precision_values, thresholds, os.path.join(result_path, "img", f'precision_recall_curve_{category}.png'))
        

import matplotlib.pyplot as plt
def draw_precision_recall_curve(recalls, precisions, thresholds, save_path,):
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions)
    #annotate the thresholds
    for i, threshold in enumerate(thresholds):
        ax.annotate(np.round(threshold, decimals=2), (recalls[i], precisions[i]))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    plt.savefig(save_path)

def main():
    if os.path.exists(RESULT_TEX):
        os.remove(RESULT_TEX)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(os.path.join(result_path, "img"), exist_ok=True)
    files = os.listdir(data_path)
    files = [f for f in files if f.endswith(".csv")] # collect files from multiple seeds or games
    file = files[0] # only one file for now
    print(f"Processing {file}")
    df = pd.read_csv(os.path.join(data_path, file), sep=";")
    df = add_contrived_columns(df)
    print(df)
    plot_ap(df)
    collect_prec_recall_data(df)

if __name__ == "__main__":
    main()