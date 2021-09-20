import pandas as pd
from utils import denormalize, get_values_for_pr_curve, get_values_for_roc_curve
import os

from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Visualizer():
    """ Visualizer wrapper based on Tensorboard.

    Returns:
        Visualizer: Class file.
    """
    def __init__(self, cfg):        
        self.cfg = cfg
        from tensorboardX import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join("../tensorboard/PaDiM/", self.cfg.name, str(self.cfg.seed), self.cfg.backbone))

    def plot_current_errors(self, total_steps, errors):
        """Plot current errros.

        Args:
            total_steps (int): Current total_steps
            errors (OrderedDict): Error for the current epoch.
        """
        self.writer.add_scalars("Loss", errors, global_step=total_steps)
        

    def plot_performance(self, epoch, performance):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            performance (OrderedDict): Performance for the current epoch.
        """
        
        self.writer.add_scalars("Performance Metrics", {k:v for k,v in performance.items() if (k != "conf_matrix" and k != "Avg Run Time (ms/batch)")}, global_step=epoch)
             
    def plot_current_conf_matrix(self, epoch, cm):
        
        def _plot_confusion_matrix(cm,
                          target_names=["Normal", "Abnormal"],
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          savefig = True):
            """
            given a sklearn confusion matrix (cm), make a nice plot

            Arguments
            ---------
            cm:           confusion matrix from sklearn.metrics.confusion_matrix

            target_names: given classification classes such as [0, 1, 2]
                        the class names, for example: ['high', 'medium', 'low']

            title:        the text to display at the top of the matrix

            cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                        see http://matplotlib.org/examples/color/colormaps_reference.html
                        plt.get_cmap('jet') or plt.cm.Blues

            normalize:    If False, plot the raw numbers
                        If True, plot the proportions

            Usage
            -----
            plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                    # sklearn.metrics.confusion_matrix
                                normalize    = True,                # show proportions
                                target_names = y_labels_vals,       # list of names of the classes
                                title        = best_estimator_name) # title of graph

            Citiation
            ---------
            http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

            """
            import matplotlib.pyplot as plt
            import numpy as np
            import itertools

            accuracy = np.trace(cm) / float(np.sum(cm))
            misclass = 1 - accuracy

            if cmap is None:
                cmap = plt.get_cmap('Blues')

            figure = plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            #plt.title(title)
            plt.colorbar()

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            if savefig:
                plt.savefig(title+".png")
            plt.close()
            return figure
        plot = _plot_confusion_matrix(cm, normalize=False, savefig=False)
        self.writer.add_figure("Confusion Matrix", plot, global_step=epoch)
        
    def plot_current_images(self, images, train_or_test="train", global_step=0):
        """ Display current images.

        Args:
            global_step (int): global step
            train_or_test (["train", "test]): Determines, which phase the model is in
            images ([FloatTensor]): [Real Image, Anomaly Map, Mask (Optional)]
        """
        self.writer.add_images("images_from_{}_step".format(str(train_or_test)), images, global_step=global_step)

    def plot_current_anomaly_map(self, image, amap, train_or_test="train", global_step=0, save_path=None, maximum_as=None):
        """ Display current images.

        Args:
            global_step (int): global step
            train_or_test (["train", "test]): Determines, which phase the model is in
            images ([FloatTensor]): [Real Image, Anomaly Map, Mask (Optional)]
        """
        
        fig, axis = plt.subplots(figsize=(4,4))
        axis.imshow(image.squeeze().permute(1, 2, 0).numpy())
        divider = make_axes_locatable(axis)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        amap_on_image = axis.imshow(amap.squeeze(), alpha=.7, cmap='viridis', vmin=0 if maximum_as else None, vmax=maximum_as)
        fig.colorbar(amap_on_image, cax=cax, orientation='vertical')
        if save_path:
            fig.savefig(save_path)
        
        
        self.writer.add_figure("images_from_{}_step".format(str(train_or_test)), fig, global_step=global_step)
        
    def plot_pr_curve(self, y_trues, y_preds, thresholds, global_step=1):    
        tpc, fpc, tnc, fnc, precisions, recalls, n_thresholds = get_values_for_pr_curve(y_trues=y_trues, y_preds=y_preds, thresholds=thresholds)
        self.writer.add_pr_curve_raw("Precision_recall_curve", true_positive_counts=tpc, false_positive_counts=fpc, true_negative_counts=tnc, false_negative_counts=fnc,
                                                precision=precisions, recall=recalls, num_thresholds=n_thresholds, global_step=global_step)
        
    def plot_histogram(self, y_trues, y_preds, threshold, global_step=1, save_path=None, tag=None):
        scores = dict()
        scores["scores"] = y_preds
        scores["labels"] = y_trues
        hist = pd.DataFrame.from_dict(scores)
        hist.to_csv(save_path if save_path else "histogram.csv")
        
        plt.ion()

            # Filter normal and abnormal scores.
        abn_scr = hist.loc[hist.labels == 1]['scores']
        nrm_scr = hist.loc[hist.labels == 0]['scores']

            # Create figure and plot the distribution.
        fig = plt.figure(figsize=(4,4))
        sns.distplot(nrm_scr, label=r'Normal Scores')
        sns.distplot(abn_scr, label=r'Abnormal Scores')
        plt.axvline(threshold, 0, 1, label='threshold', color="red")
        plt.legend()
        plt.yticks([])
        plt.xlabel(r'Anomaly Scores')
        self.writer.add_figure(tag if tag else "Histogram", fig, global_step)
        
    def plot_roc_curve(self, y_trues, y_preds, global_step=1, tag=None):
        fpr, tpr, roc_auc = get_values_for_roc_curve(y_trues, y_preds)
        fig = plt.figure(figsize=(4,4))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        self.writer.add_figure(tag if tag else "ROC-Curve", fig, global_step)