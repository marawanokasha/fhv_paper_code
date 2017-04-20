import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import coverage_error
import matplotlib.pylab as plt

"""
Utils for calculating metrics
"""


get_binary = lambda x: 1 if x > 0 else 0
get_binary = np.vectorize(get_binary)

get_binary_0_5 = lambda x: 1 if x > 0.5 else 0
get_binary_0_5 = np.vectorize(get_binary_0_5)


def get_top_N_percentage(y_score, y_true, max_N=3):
    """
    Get percentage of correct labels that are in the top N scores
    """
    num_all_true = 0
    num_found_in_max_N = 0
    for i in xrange(y_score.shape[0]):
        y_score_row = y_score[i,:]
        y_true_row = y_true[i,:]
        desc_score_indices = np.argsort(y_score_row)[::-1]
        true_indices = np.where(y_true_row ==1)[0]

        num_true_in_row = len(true_indices)
        num_all_true += num_true_in_row
        for i, score_index in enumerate(desc_score_indices):
            # only iterate through the score list till depth N, but make sure you also account for the case where
            # the number of true labels for the current row is higher than N
            if i >= max_N and i >= num_true_in_row:
                break
            if score_index in true_indices:
                num_found_in_max_N += 1
    return float(num_found_in_max_N)/ num_all_true


def get_metrics(y_true, y_score, y_binary_score):
    """
    create the metrics object containing all relevant metrics
    """
    metrics = {}
    metrics['total_positive'] = np.sum(np.sum(y_binary_score))
    #TODO remove those two when running on the whole set to avoid excessive storage costs
    #metrics['y_true'] = y_true
    #metrics['y_score'] = y_score
    #metrics['y_binary_score'] = y_binary_score
    metrics['coverage_error'] = coverage_error(y_true, y_score)
    metrics['average_num_of_labels'] = round(float(np.sum(np.sum(y_true, axis=1)))/y_true.shape[0], 2)
    #metrics['average_precision_micro'] = sklearn.metrics.average_precision_score(y_true, y_binary_score, average='micro')
    #metrics['average_precision_macro'] = sklearn.metrics.average_precision_score(y_true, y_binary_score, average='macro')
    metrics['precision_micro'] = sklearn.metrics.precision_score(y_true, y_binary_score, average='micro')
    metrics['precision_macro'] = sklearn.metrics.precision_score(y_true, y_binary_score, average='macro')
    metrics['recall_micro'] = sklearn.metrics.recall_score(y_true, y_binary_score, average='micro')
    metrics['recall_macro'] = sklearn.metrics.recall_score(y_true, y_binary_score, average='macro')
    metrics['f1_micro'] = sklearn.metrics.f1_score(y_true, y_binary_score, average='micro')
    metrics['f1_macro'] = sklearn.metrics.f1_score(y_true, y_binary_score, average='macro')

    # only calculate those for cases with a small number of labels (sections only)
    if y_true.shape[1] < 100:
        precision_scores = np.zeros(y_true.shape[1])
        for i in range(0, y_true.shape[1]):
            precision_scores[i] = sklearn.metrics.precision_score(y_true[:,i], y_binary_score[:,i])
        metrics['precision_scores_array'] = precision_scores.tolist()

        recall_scores = np.zeros(y_true.shape[1])
        for i in range(0, y_true.shape[1]):
            recall_scores[i] = sklearn.metrics.recall_score(y_true[:,i], y_binary_score[:,i])
        metrics['recall_scores_array'] = recall_scores.tolist()

        f1_scores = np.zeros(y_true.shape[1])
        for i in range(0, y_true.shape[1]):
            f1_scores[i] = sklearn.metrics.f1_score(y_true[:,i], y_binary_score[:,i])
        metrics['f1_scores_array'] = f1_scores.tolist()

    metrics['top_1'] = get_top_N_percentage(y_score, y_true, max_N=1)
    metrics['top_3'] = get_top_N_percentage(y_score, y_true, max_N=3)
    metrics['top_5'] = get_top_N_percentage(y_score, y_true, max_N=5)

    return metrics


def get_formatted_multilabel_confusion_matrix(y_true, y_pred, labels = []):
    """
    labels: string labels to use for the classes represented by the columns in y_true and y_pred, should be in the same
            order as the classes of the columns
    """
    if len(labels) != y_true.shape[1]:
        raise "labels must have the same size as the number of columns in y_true"

    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)

    data_frame = pd.DataFrame(data=confusion_matrix, index=labels + ["None"], columns=labels + ["None"])

    return data_frame


def multilabel_confusion_matrix(y_true, y_pred):
    """
    y_true: 2d array with size (num of samples x num of classes)
    y_pred: 2d array with size (num of samples x num of classes)
    """
    if len(y_true.shape) != 2 or len(y_pred.shape) != 2:
        raise "input arrays must be two dimensional binary matrices"

    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[1] != y_pred.shape[1]:
        raise "y_true and y_pred should be of the same size"


    # the +1 is for the "None" label
    num_classes = y_true.shape[1]
    confusion_matrix = np.zeros((num_classes+1, num_classes +1))
    none_label_index = num_classes

    for row_index, row in enumerate(y_true):

        true_indices = set(np.nonzero(row)[0])
        pred_indices = set(np.nonzero(y_pred[row_index])[0])

        tp_indices = true_indices & pred_indices

        for tp_index in tp_indices:
            confusion_matrix[tp_index, tp_index] += 1

        fn_indices = true_indices - tp_indices
        fp_indices = pred_indices - tp_indices


        i = 0
        for fn_index in sorted(fn_indices):
            if len(fp_indices) > i:
                matching_fp_index = sorted(fp_indices)[i]
                confusion_matrix[fn_index, matching_fp_index] += 1
                fp_indices.remove(matching_fp_index)
                i += 1
            else:
                break

        # get all the fn indices that were not exhausted in the previous loop
        fn_indices = sorted(fn_indices)[i:]

        for fn_index in fn_indices:
            confusion_matrix[fn_index, none_label_index] += 1

        for fp_index in fp_indices:
            confusion_matrix[none_label_index, fp_index] += 1

    return confusion_matrix


class MetricsGraph:
    def __init__(self):
        self.coverage_errors = []
        self.average_num_labels = []

        self.f1_micros = []
        self.precision_micros = []
        self.recall_micros = []
        self.f1_macros = []
        self.precision_macros = []
        self.recall_macros = []

        self.top_1s = []
        self.top_3s = []
        self.top_5s = []

        self.epochs = []

        self.fig = None
        self.ax = None
        self.ax2 = None

        self.coverage_error_max = None

    def init_graph(self, coverage_error_max=10):
        self.fig = plt.figure(figsize=(12,6), dpi=80)
        self.ax = plt.subplot(121)
        self.ax2 = plt.subplot(122)
        self.fig.subplots_adjust(top=0.72, bottom=0.1, left=0.05, right=0.95)
        self.ax.set_xlabel("Epochs")
        self.ax2.set_xlabel("Epochs")
        self.coverage_error_max = coverage_error_max

    def _add_metrics(self, metrics, epoch):
        self.coverage_errors.append(metrics['coverage_error'])
        self.average_num_labels.append(metrics['average_num_of_labels'])

        self.f1_micros.append(metrics['f1_micro'])
        self.precision_micros.append(metrics['precision_micro'])
        self.recall_micros.append(metrics['recall_micro'])
        self.f1_macros.append(metrics['f1_macro'])
        self.precision_macros.append(metrics['precision_macro'])
        self.recall_macros.append(metrics['recall_macro'])

        self.top_1s.append(metrics['top_1']  if 'top_1' in metrics else get_top_N_percentage(metrics['y_score'], metrics['y_true'], max_N=1))
        self.top_3s.append(metrics['top_3']  if 'top_3' in metrics else get_top_N_percentage(metrics['y_score'], metrics['y_true'], max_N=3))
        self.top_5s.append(metrics['top_5']  if 'top_5' in metrics else get_top_N_percentage(metrics['y_score'], metrics['y_true'], max_N=5))

        self.epochs.append(epoch)

    def add_metrics_to_graph(self, metrics, epoch, draw_now=True):

        self._add_metrics(metrics, epoch)
        if draw_now:
            self.draw()

    def draw(self):

        first_epoch = self.epochs[0]
        last_epoch = self.epochs[-1]

        coverage_error_line, = self.ax.plot(self.epochs, self.coverage_errors, 'r-', label='Coverage Error')
        average_num_labels_line, = self.ax.plot(self.epochs, self.average_num_labels, 'g-', label='Avg Num. of Labels')

        self.ax.legend(handles=[coverage_error_line, average_num_labels_line],
                  bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        self.ax.axis([first_epoch, last_epoch, 0, self.coverage_error_max])
        # show the average number of labels as a separate y-tick
        curr_min_cov = [min(self.coverage_errors)]
        prev_min_cov = [min(self.coverage_errors[:-1])] if len(self.coverage_errors) > 1 else []
        self.ax.set_yticks(list(set(self.ax.get_yticks())- set(prev_min_cov) ) + [self.average_num_labels[0]] + curr_min_cov)

        f1_micro_line, = self.ax2.plot(self.epochs, self.f1_micros, 'g-', label='F1 Micro')
        precision_micro_line, = self.ax2.plot(self.epochs, self.precision_micros, 'r-', label='Precision Micro')
        recall_micro_line, = self.ax2.plot(self.epochs, self.recall_micros, 'b-', label='Recall Micro')
        f1_macro_line, = self.ax2.plot(self.epochs, self.f1_macros, 'g--', label='F1 Macro')
        precision_macro_line, = self.ax2.plot(self.epochs, self.precision_macros, 'r--', label='Precision Macro')
        recall_macro_line, = self.ax2.plot(self.epochs, self.recall_macros, 'b--', label='Recall Macro')

        top_1_line, = self.ax2.plot(self.epochs, self.top_1s, 'g-.', label='Top 1 %')
        top_3_line, = self.ax2.plot(self.epochs, self.top_3s, 'r-.', label='Top 3 %')
        top_5_line, = self.ax2.plot(self.epochs, self.top_5s, 'b-.', label='Top 5 %')

        self.ax2.legend(handles=[f1_micro_line, precision_micro_line, recall_micro_line,
                                 f1_macro_line, precision_macro_line, recall_macro_line,
                                 top_1_line, top_3_line, top_5_line],
                  bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        self.ax2.axis([first_epoch, last_epoch, 0, 1])
        curr_max_f1_micro = [max(self.f1_micros)]
        prev_max_f1_micro = [max(self.f1_micros[:-1])] if len(self.f1_micros) > 1 else []
        self.ax2.set_yticks(list(set(self.ax2.get_yticks())- set(prev_max_f1_micro) ) + curr_max_f1_micro)

        self.fig.canvas.draw()


if __name__ ==  "__main__":

    true = np.array([[1,0,0], [1,0,1], [0,1,0], [0,1,0], [0,0,1], [0,1,1]])
    pred = np.array([[0,1,0], [1,0,0], [0,0,1], [0,1,0], [1,0,1], [1,0,1]])
    labels = ["A", "B", "C"]

    confusion_matrix = get_formatted_multilabel_confusion_matrix(true, pred, labels)

    print confusion_matrix



