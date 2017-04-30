import json
import nltk
from nltk.tokenize import RegexpTokenizer
import string
import math
import os
import io
import sys
import time
from collections import namedtuple
import cPickle as pickle
from multiprocessing import Queue, Process
import gzip
from ExtendedPVDocumentBatchGenerator import BatchWrapper

import numpy as np
import random

import itertools

from sklearn.metrics import coverage_error
import sklearn.metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn.preprocessing import MultiLabelBinarizer

from gensim.models.doc2vec import Doc2Vec, LabeledSentence

import logging
from logging import info
from functools import partial

root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # adds a default StreamHanlder

# specify on command line which vocabs are to be created so that we can run multiple instances from command line
# in parallel

if len(sys.argv) < 3:
    print('please specify level and item')
    sys.exit(1)

known_items = ('document', 'claims', 'abstract', 'description')

if sys.argv[2] not in ('document', 'claims', 'abstract', 'description'):
    print("Unknown item; pelase use:")
    print(known_items)
    sys.exit(1)

level = int(sys.argv[1])
model_name = sys.argv[2]

SVM_SEED = 1234
DOC2VEC_SEED = 1234
MIN_WORD_COUNT = 100
NUM_CORES = 16
GLOBAL_VARS = namedtuple('GLOBAL_VARS', ['MODEL_NAME', 'DOC2VEC_MODEL_NAME', 'DOC2VEC_MODEL', 
                                         'SVM_MODEL_NAME', 'NN_MODEL_NAME'])
VOCAB_MODEL = "vocab_model"
MODEL_PREFIX = "model"
VALIDATION_MINI_BATCH_SIZE = 10000
VALIDATION_DICT = "validation_dict.pkl.gz"
TEST_MATRIX = "test_matrix.pkl"
TEST_DICT = "test_dict.pkl"
METRICS = "metrics.pkl"
CLASSIFIER = "classifier.pkl"
TYPE_CLASSIFIER = "{}_classifier.pkl"

DOC2VEC_SIZE = 200
DOC2VEC_WINDOW = 2
DOC2VEC_MAX_VOCAB_SIZE = None
DOC2VEC_SAMPLE = 1e-3
DOC2VEC_TYPE = 1
DOC2VEC_HIERARCHICAL_SAMPLE = 0
DOC2VEC_NEGATIVE_SAMPLE_SIZE = 10
DOC2VEC_CONCAT = 0
DOC2VEC_MEAN = 1
DOC2VEC_TRAIN_WORDS = 0
DOC2VEC_EPOCHS = 1  # we do our training manually one epoch at a time
DOC2VEC_MAX_EPOCHS = 8
REPORT_DELAY = 20  # report the progress every x seconds
REPORT_VOCAB_PROGRESS = 10000  # report vocab progress every x documents
VALIDATION_MINI_BATCH_SIZE = 10000

root_location = "/mnt/virtual-machines/data/"
word2vec_questions_file = result = root_location + 'tensorflow/word2vec/questions-words.txt'

preprocessed_location = result = root_location + "preprocessed_data/extended_pv_abs_desc_claims_full_chunks/"

training_preprocessed_files_prefix = preprocessed_location + "extended_pv_training_docs_data_preprocessed-"
validation_preprocessed_files_prefix = preprocessed_location + "extended_pv_validation_docs_data_preprocessed-"
test_preprocessed_files_prefix = preprocessed_location + "extended_pv_test_docs_data_preprocessed-"


def ensure_disk_location_exists(location):
    if not os.path.exists(location):
        os.makedirs(location)

# loading vocabulary if exists
doc2vec_model_save_location = os.path.join(root_location,
                                           "parameter_search_doc2vec_models_" + str(level) + '_' + model_name,
                                           "full")
if not os.path.exists(doc2vec_model_save_location):
    os.makedirs(doc2vec_model_save_location)
if not os.path.exists(os.path.join(doc2vec_model_save_location, VOCAB_MODEL)):
    os.makedirs(os.path.join(doc2vec_model_save_location, VOCAB_MODEL))

placeholder_model_name = 'doc2vec_size_{}_w_{}_type_{}_concat_{}_mean_{}_trainwords_{}_hs_{}_neg_{}_vocabsize_{}_model_{}'.format(DOC2VEC_SIZE,
                                                                DOC2VEC_WINDOW,
                                                                'dm' if DOC2VEC_TYPE == 1 else 'pv-dbow',
                                                                DOC2VEC_CONCAT, DOC2VEC_MEAN,
                                                                DOC2VEC_TRAIN_WORDS,
                                                                DOC2VEC_HIERARCHICAL_SAMPLE,DOC2VEC_NEGATIVE_SAMPLE_SIZE,
                                                                str(DOC2VEC_MAX_VOCAB_SIZE),
                                                                str(level) + '_' + model_name
                                                                )

GLOBAL_VARS.DOC2VEC_MODEL_NAME = placeholder_model_name
placeholder_model_name = os.path.join(placeholder_model_name, "epoch_{}")
info("FILE " + os.path.join(doc2vec_model_save_location, VOCAB_MODEL, MODEL_PREFIX))
doc2vec_model = Doc2Vec(size=DOC2VEC_SIZE, window=DOC2VEC_WINDOW, min_count=MIN_WORD_COUNT,
                        max_vocab_size=DOC2VEC_MAX_VOCAB_SIZE,
                        sample=DOC2VEC_SAMPLE, seed=DOC2VEC_SEED, workers=NUM_CORES,
                        # doc2vec algorithm dm=1 => PV-DM, dm=2 => PV-DBOW, PV-DM dictates CBOW for words
                        dm=DOC2VEC_TYPE,
                        # hs=0 => negative sampling, hs=1 => hierarchical softmax
                        hs=DOC2VEC_HIERARCHICAL_SAMPLE, negative=DOC2VEC_NEGATIVE_SAMPLE_SIZE,
                        dm_concat=DOC2VEC_CONCAT,
                        # would train words with skip-gram on top of cbow, we don't need that for now
                        dbow_words=DOC2VEC_TRAIN_WORDS,
                        iter=DOC2VEC_EPOCHS
                        )

if not os.path.exists(os.path.join(doc2vec_model_save_location, VOCAB_MODEL, MODEL_PREFIX)):
    info("creating vocabular for " + str(level) + ' ' + model_name + ' in ')
    training_docs_iterator = BatchWrapper(training_preprocessed_files_prefix, batch_size=10000, level=level,
                                          level_type=model_name)
    doc2vec_model.build_vocab(sentences=training_docs_iterator, progress_per=REPORT_VOCAB_PROGRESS)
    doc2vec_model.save(os.path.join(doc2vec_model_save_location, VOCAB_MODEL, MODEL_PREFIX))
else:
    info("loading " + os.path.join(doc2vec_model_save_location, VOCAB_MODEL, MODEL_PREFIX))
    doc2vec_vocab_model = Doc2Vec.load(
        os.path.join(doc2vec_model_save_location, VOCAB_MODEL, MODEL_PREFIX)
    )
    doc2vec_model.reset_from(doc2vec_vocab_model)

doc2vec_model.alpha = 0.025
doc2vec_model.min_alpha = 0.025
DOC2VEC_ALPHA_DECREASE = 0.001
epoch_validation_metrics = []
epoch_training_metrics = []
epoch_word2vec_metrics = []
epoch = 0
start_epoch = 1

for epoch in range(1, DOC2VEC_MAX_EPOCHS + 1):
    GLOBAL_VARS.MODEL_NAME = placeholder_model_name.format(epoch)
    if os.path.exists(os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME, MODEL_PREFIX)):
        start_epoch = epoch

if start_epoch > 1:
    info('resuming from epoch ' + str(start_epoch))
    GLOBAL_VARS.MODEL_NAME = placeholder_model_name.format(start_epoch)
    # if a model of that epoch already exists, we load it and proceed to the next epoch
    doc2vec_model = Doc2Vec.load(os.path.join(
        doc2vec_model_save_location,
        GLOBAL_VARS.MODEL_NAME,
        MODEL_PREFIX))
    start_epoch += 1


for epoch in range(start_epoch, DOC2VEC_MAX_EPOCHS + 1):
    # set new filename/path to include the epoch
    GLOBAL_VARS.MODEL_NAME = placeholder_model_name.format(epoch)
    info("****************** Epoch {} --- Working on {} *******************".format(epoch, GLOBAL_VARS.MODEL_NAME))
    # train the doc2vec model
    training_docs_iterator = BatchWrapper(training_preprocessed_files_prefix, batch_size=10000, level=level,
                                          level_type=model_name)
    doc2vec_model.train(sentences=training_docs_iterator, report_delay=REPORT_DELAY)
    doc2vec_model.alpha -= DOC2VEC_ALPHA_DECREASE  # decrease the learning rate
    doc2vec_model.min_alpha = doc2vec_model.alpha  # fix the learning rate, no decay
    ensure_disk_location_exists(os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME))
    doc2vec_model.save(os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME, MODEL_PREFIX))


if epoch != 8:
    print("still training epochs missing: " + str(epoch))
    sys.exit(1)


class DocReader(Process):
    def __init__(self, level, level_type, preprocessed_files_prefix, out_queue, num_reader):
        super(DocReader, self).__init__()
        self.out_queue = out_queue
        self.num_reader = num_reader
        self.inference_docs_iterator = BatchWrapper(
            preprocessed_files_prefix,
            batch_size=None,
            buffer_size=10000,
            level=level,
            level_type=level_type)

    def run(self):
        while True:
            for item in self.inference_docs_iterator:
                self.out_queue.put(item)
            for i in range(0, self.num_reader):
                self.out_queue.put(False, block=True, timeout=None)
            sys.exit()


class DocInferer(Process):

    def __init__(self, model, in_queue, out_queue):
        super(DocInferer, self).__init__()
        self.model = model
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            doc_tuple = self.in_queue.get(block=True)
            if doc_tuple is False:
                self.out_queue.put(False)
                sys.exit()
            self.out_queue.put((doc_tuple[0], self.model.infer_vector(doc_tuple[1])))


def get_extended_docs_with_inference_data_only(doc2vec_model, file_to_write, preprocessed_files_prefix, level, level_type):
    """
    Use the trained doc2vec model to get the paragraph vector representations of the validation or test documents
    """
    if os.path.exists(os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME, file_to_write)):
        info("===== Loading inference vectors")
        info(os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME, file_to_write))
        inference_documents_reps = pickle.load(io.BufferedReader(gzip.open(os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME, file_to_write))))
        info("Loaded inference vectors matrix")
    else:
        inference_documents_reps = {}
        info("===== Getting vectors with inference")
        processes = []
        out_queue = Queue(maxsize=10000)
        in_queue = Queue(maxsize=10000)
        reader = DocReader(level, level_type, preprocessed_files_prefix, in_queue, num_reader=NUM_CORES)
        reader.start()
        for process_id in range(1, NUM_CORES + 1):
            processes.append(DocInferer(doc2vec_model, in_queue=in_queue, out_queue=out_queue))
        for p in processes:
            p.start()
        proc_finished = 0
        while True:
            item = out_queue.get(block=True)
            if item is False:
                proc_finished += 1
            else:
                inference_documents_reps[item[0]] = item[1]
            if proc_finished == NUM_CORES:
                info("===== All processes terminated")
                break
        # the infered vectors should be small enough to be kept in memory for the whole validation set
        pickle.dump(inference_documents_reps,
                    io.BufferedWriter(gzip.open(os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME, file_to_write), 'w')))
        info('storing in: ' + os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME, file_to_write))
    return inference_documents_reps

Xv = get_extended_docs_with_inference_data_only(doc2vec_model, VALIDATION_DICT, validation_preprocessed_files_prefix,
                                                level=level,
                                                level_type=model_name)
Xt = get_extended_docs_with_inference_data_only(doc2vec_model, TEST_DICT, test_preprocessed_files_prefix,
                                                level=level,
                                                level_type=model_name)

