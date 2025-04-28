import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap='Blues', show=True, save_path=None):
    """
    Plots a confusion matrix using matplotlib and seaborn.

    Parameters:
    cm (array-like): Confusion matrix data.
    labels (list): List of labels for the confusion matrix.
    title (str): Title of the plot.
    cmap (str): Colormap to use for the heatmap.

    Returns:
    None
    """


    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()