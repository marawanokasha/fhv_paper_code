import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from logging import info
from multiprocessing import Queue, Process

import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model


sys.path.append(os.path.abspath('..'))
from utils.metrics import get_metrics, get_binary_0_5


# ranges for learning graph shown
metrics_graph_ranges = {
    'sections': {'min':0, 'max': 0.5},
    'classes': {'min':0, 'max': 0.05},
    'subclasses': {'min':0, 'max': 0.05}
}

QUEUE_SIZE = 100

TRAINING_DATA_MATRIX = "X_level_{}.npy"
TRAINING_LABELS_MATRIX = "y_{}.npy"
VALIDATION_DATA_MATRIX = "Xv_level_{}.npy"
VALIDATION_LABELS_MATRIX = "yv_{}.npy"
TEST_DATA_MATRIX = "Xt_level_{}.npy"
TEST_LABELS_MATRIX = "yt_{}.npy"


data_type_file_dict ={
    "training": TRAINING_DATA_MATRIX,
    "validation": VALIDATION_DATA_MATRIX,
    "test": TEST_DATA_MATRIX,
}
labels_type_file_dict ={
    "training": TRAINING_LABELS_MATRIX,
    "validation": VALIDATION_LABELS_MATRIX,
    "test": TEST_LABELS_MATRIX,
}

def get_data_dirs(base_location, classifications_type, level, data_type):
    data_dir = os.path.join(base_location, data_type_file_dict[data_type].format(level))
    labels_dir = os.path.join(base_location, labels_type_file_dict[data_type].format(classifications_type))
    return data_dir, labels_dir

def get_data(data_file, labels_file, mmap=False):
    mmap_mode = None
    if mmap == True:
        mmap_mode = "r"
    X_data = np.load(data_file, mmap_mode=mmap_mode)
    y_data = np.load(labels_file, mmap_mode=mmap_mode)
    return X_data, y_data


class MetricsCallback(keras.callbacks.Callback):
    """
    Callback called by keras after each epoch. Records the best validation loss and periodically checks the
    validation metrics
    """
    def __init__(self, classifications_type, level, batch_size, is_mlp=False):
        MetricsCallback.EPOCHS_BEFORE_VALIDATION = 10
        MetricsCallback.GRAPH_MIN = metrics_graph_ranges[classifications_type]['min']
        MetricsCallback.GRAPH_MAX = metrics_graph_ranges[classifications_type]['max']
        self.classifications_type = classifications_type
        self.level = level
        self.batch_size = batch_size
        self.is_mlp = is_mlp

    def on_train_begin(self, logs={}):
        self.epoch_index = 0
        self.val_loss_reductions = 0
        self.metrics_dict = {}
        self.best_val_loss = np.iinfo(np.int32).max
        self.best_weights = None
        self.best_validation_metrics = None

        self.losses = []
        self.val_losses = []
        self.fig = plt.figure(figsize=(12,6), dpi=80)
        self.ax = plt.subplot(111)
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_index += 1
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        loss_line, = self.ax.plot(range(1,self.epoch_index+1), self.losses, 'g-', label='Training Loss')
        val_loss_line, = self.ax.plot(range(1,self.epoch_index+1), self.val_losses, 'r-', label='Validation Loss')
        self.ax.legend(handles=[loss_line, val_loss_line])
        self.ax.set_ylim((MetricsCallback.GRAPH_MIN, MetricsCallback.GRAPH_MAX))
        self.fig.canvas.draw()
        if logs['val_loss'] < self.best_val_loss:
            self.val_loss_reductions += 1
            self.best_val_loss = logs['val_loss']
            self.best_weights = self.model.get_weights()
            print '\r    \r' # to remove the previous line of verbose output of model fit
            #time.sleep(0.1)
            info('Found lower val loss for epoch {} => {}'.format(self.epoch_index, round(logs['val_loss'], 5)))
            if self.val_loss_reductions % MetricsCallback.EPOCHS_BEFORE_VALIDATION == 0:

                info('Validation Loss Reduced {} times'.format(self.val_loss_reductions))
                info('Evaluating on Validation Data')
                Xv_file, yv_file = get_data_dirs(self.classifications_type, self.level, 'validation')
                Xv, yv = get_data(Xv_file, yv_file, mmap=True)
                yvp = self.model.predict_generator(generator=batch_generator(self.classifications_type, self.level,
                                                   self.batch_size, is_mlp=self.is_mlp, validate=True),\
                                                   max_q_size=QUEUE_SIZE,\
                                                   val_samples=yv.shape[1])
                yvp_binary = get_binary_0_5(yvp)
                info('Generating Validation Metrics')
                validation_metrics = get_metrics(yv, yvp, yvp_binary)
                print "****** Validation Metrics: Cov Err: {:.3f} | Top 3: {:.3f} | Top 5: {:.3f} | F1 Micro: {:.3f} | F1 Macro: {:.3f}".format(
                    validation_metrics['coverage_error'], validation_metrics['top_3'], validation_metrics['top_5'],
                    validation_metrics['f1_micro'], validation_metrics['f1_macro'])
                self.metrics_dict[self.epoch_index] = validation_metrics
#                 self.best_validation_metrics = validation_metrics


class ArrayReader(Process):
    def __init__(self, input_file, label_file, out_queue, batch_size, is_mlp=False, validate=False):
        super(ArrayReader, self).__init__()
        self.is_mlp = is_mlp
        self.validate = validate
        self.q = out_queue
        self.batch_size = batch_size
        self.input_file = input_file
        self.label_file = label_file

    def run(self):
        x_file = np.load(self.input_file, mmap_mode='r')
        y_file = np.load(self.label_file, mmap_mode='r')
        start_item = 0
        num_iter = 0
        while True:
            if start_item > y_file.shape[0]:
                info('in new epoch for {}'.format(os.path.basename(self.input_file)))
                start_item = 0
            y_batch = y_file[start_item: start_item + self.batch_size]
            x_batch = x_file[start_item: start_item + self.batch_size]
            if self.is_mlp:
                x_batch = np.reshape(x_batch, (x_batch.shape[0], x_batch.shape[1] * x_batch.shape[2]))
            start_item += self.batch_size
            num_iter += 1
            try:
                self.q.put((x_batch, y_batch), block=True)
            except:
                return


def batch_generator(input_file, label_file, batch_size, is_mlp=False, validate=False):
    q = Queue(maxsize=QUEUE_SIZE)
    p = ArrayReader(input_file, label_file, q, batch_size, is_mlp, validate)
    p.start()
    while True:
        item = q.get()
        if not item:
            p.terminate()
            info('Finished batch iteration')
            raise StopIteration()
        else:
            yield item