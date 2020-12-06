from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
def plotConfusionMatrix(mat,CLASSES):
    # Normalise
    # normalized_mat = mat.astype('float') /mat.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(mat,index=CLASSES,columns=CLASSES)
    plt.figure(figsize=(12,12))
    sn.set(font_scale=1)  # for label size
    sn.heatmap(df_cm, annot=True,fmt='d', cbar=False)  # font size
    plt.show()


def plot_images(images, cls_true, cls_pred=None):
    # Create figure with 3x3 sub-plots.
    plt.figure(figsize=(12, 12))
    count = len(images)
    fig, axes = plt.subplots(int(sqrt(count)), int(sqrt(count)),figsize=(20, 20))


    #     fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i])
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        # Remove ticks from the plot.
        # ax.set_xticks([])
        # ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.tight_layout()
    plt.show()

