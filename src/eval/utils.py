import torch
import numpy as np
import matplotlib.pyplot as plt


def flatten(nested_list):
    return torch.cat([e for lst in nested_list for e in lst])


def plot_confusion_matrix(cm, label_list, path, title="Confusion Matrix"):
    """
    Compute the confusion matrix for a given classifier
    :param clf: classifier
    :param inputs: inputs
    :param labels: labels
    :return: confusion matrix
    """
    # plot with matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # also write value in each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # plt.text(j, i, "{:0.2f}".format(cm[i, j]),
            #         horizontalalignment="center",
            #         color="white" if cm[i, j] > thresh else "black")
            # format as integer
            plt.text(j, i, "{:d}".format(cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.title(title)
    #plt.colorbar()
    plt.xticks(np.arange(len(label_list)), label_list, rotation=45)
    plt.yticks(np.arange(len(label_list)), label_list)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

