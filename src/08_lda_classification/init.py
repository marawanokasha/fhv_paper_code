import os
import sys
import argparse
import random
import cPickle as pickle

from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model

import logging
from logging import info


sys.path.append(os.path.abspath('..'))
from utils.metrics import get_metrics
from utils.classification import get_label_data
from utils.file import ensure_disk_location_exists

root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # adds a default StreamHanlder


RANDOM_SEED = 10000
random.seed(RANDOM_SEED)

SVM_SEED = 1234

CLASSIFIER_FILE = '{}_classifier.pkl'
VALIDATION_METRICS_FILENAME= '{}_validation_metrics.pkl'
TRAINING_METRICS_FILENAME = '{}_training_metrics.pkl'
TEST_METRICS_FILENAME = '{}_test_metrics.pkl'


root_location = "../../data/"
exports_location = root_location + "exported_data/"
lda_location = root_location + "extended_pv_lda/"
svm_location = root_location + "svm_lda_results/"

classifications_index_file = os.path.join(exports_location, "classifications_index.pkl")
doc_classification_map_file = os.path.join(exports_location, "doc_classification_map.pkl")
sections_file = os.path.join(exports_location, "sections.pkl")
valid_classes_file = os.path.join(exports_location, "valid_classes.pkl")
valid_subclasses_file = os.path.join(exports_location, "valid_subclasses.pkl")
classifications_file = os.path.join(exports_location, "classifications.pkl")
doc_lengths_map_file = os.path.join(exports_location, "doc_lengths_map.pkl")
training_docs_list_file = os.path.join(exports_location, "training_docs_list.pkl")
validation_docs_list_file = os.path.join(exports_location, "validation_docs_list.pkl")
test_docs_list_file = os.path.join(exports_location, "test_docs_list.pkl")


## Load utility data

doc_classification_map = pickle.load(open(doc_classification_map_file))
sections = pickle.load(open(sections_file))
valid_classes = pickle.load(open(valid_classes_file))
valid_subclasses = pickle.load(open(valid_subclasses_file))
training_docs_list = pickle.load(open(training_docs_list_file))
validation_docs_list = pickle.load(open(validation_docs_list_file))
test_docs_list = pickle.load(open(test_docs_list_file))



classification_types = {
    "sections": sections,
    "classes": valid_classes,
    "subclasses": valid_subclasses
}
possible_data_types = ["tf", "sublinear_tf", "tf_idf", "sublinear_tf_idf","bm25"]

# specify on command line the SVM parameters and the bow representation to use

parser = argparse.ArgumentParser(description='Run SVM on BOW data')
parser.add_argument("-c", "--classificationsType", choices=classification_types.keys(), required=True)
parser.add_argument("-d", "--dataType", choices=possible_data_types, required=True)
parser.add_argument("-r", "--svmReg", type=float, required=True)
parser.add_argument("-i", "--svmIterations", type=int, required=True)
parser.add_argument("-w", "--svmClassWeights", type=int, default=None, help="set whether to use biased class weights (use \"balanced\" value) or no bias (leave empty)")
args = parser.parse_args()

print args

SVM_ITERATIONS = args.svmIterations
SVM_REG = args.svmReg
SVM_CLASS_WEIGHTS = args.svmClassWeights
classifications_type = args.classificationsType
classifications = classification_types[classifications_type]
data_type = args.dataType
SVM_MODEL_NAME = 'svm_iter_{}_reg_{}_classweights_{}'.format(SVM_ITERATIONS, SVM_REG, str(SVM_CLASS_WEIGHTS))

LDA_TOPICS = 1000
LDA_ITERATIONS = 50
LDA_BATCH_SIZE = 4096
LDA_DECAY = 0.5
LDA_EVALUATE_EVERY = 1000
LDA_VERBOSE = 2
LDA_LEARNING_METHOD = 'online'
LDA_MODEL_NAME = "lda_{}_topics_{}_iter_{}_batch_{}_decay_{}_evaluate-every_{}".format(LDA_LEARNING_METHOD,
                                                                                       LDA_TOPICS, LDA_ITERATIONS,
                                                                                       LDA_BATCH_SIZE, LDA_DECAY,
                                                                                       LDA_EVALUATE_EVERY)

info("=============== Classifying using data: {} ================".format(data_type))

data_training_location = os.path.join(lda_location, LDA_MODEL_NAME, data_type, "lda_training_data.pkl")
data_training_docids_location = os.path.join(exports_location, "{}_training_sparse_docids.pkl".format(data_type))
data_validation_location = os.path.join(lda_location, LDA_MODEL_NAME, data_type, "lda_validation_data.pkl")
data_validation_docids_location = os.path.join(exports_location, "{}_validation_sparse_docids.pkl".format(data_type))
data_test_location = os.path.join(lda_location, LDA_MODEL_NAME, data_type, "lda_test_data.pkl")
data_test_docids_location = os.path.join(exports_location, "{}_test_sparse_docids.pkl".format(data_type))


# Get the training data
info('Getting Training Data')
X = pickle.load(open(data_training_location, "r"))
training_data_docids = pickle.load(open(data_training_docids_location, "r"))
y = get_label_data(classifications, training_data_docids, doc_classification_map)

info('Training Classifier')
clf = OneVsRestClassifier(linear_model.SGDClassifier(loss='hinge', penalty='l2',
                                                     #alpha is the 1/C parameter
                                                     alpha=SVM_REG, fit_intercept=True, n_iter=SVM_ITERATIONS,
                                                     #n_jobs=-1 means use all cpus
                                                     shuffle=True, verbose=0, n_jobs=1,
                                                     #eta0 is the learning rate when we use constant configuration
                                                     random_state=SVM_SEED, learning_rate='optimal', eta0=0.0,
                                                     class_weight=SVM_CLASS_WEIGHTS, warm_start=False), n_jobs=1)
clf.fit(X,y)

# Training Metrics
info('Evaluating on Training Data')
yp = clf.predict(X)
yp_score = clf.decision_function(X)
info('Calculating training metrics')
training_metrics = get_metrics(y, yp_score, yp)
print "** Training Metrics: Cov Err: {:.3f}, Avg Labels: {:.3f}, \n\t\t Top 1: {:.3f}, Top 3: {:.3f}, Top 5: {:.3f}, \n\t\t F1 Micro: {:.3f}, F1 Macro: {:.3f}, Total Pos: {:,d}".format(
    training_metrics['coverage_error'], training_metrics['average_num_of_labels'],
    training_metrics['top_1'], training_metrics['top_3'], training_metrics['top_5'],
    training_metrics['f1_micro'], training_metrics['f1_macro'], training_metrics['total_positive'])

# Get the validation data
info('Getting Valdiation Data')
Xv = pickle.load(open(data_validation_location,'r'))
validation_data_docids = pickle.load(open(data_validation_docids_location, "r"))
yv = get_label_data(classifications, validation_data_docids, doc_classification_map)

# Validation Metrics
info('Evaluating on Validation Data')
yvp = clf.predict(Xv)
yvp_score = clf.decision_function(Xv)
validation_metrics = get_metrics(yv, yvp_score, yvp)
print "** Validation Metrics: Cov Err: {:.3f}, Avg Labels: {:.3f}, \n\t\t Top 1: {:.3f}, Top 3: {:.3f}, Top 5: {:.3f}, \n\t\t F1 Micro: {:.3f}, F1 Macro: {:.3f}, Total Pos: {:,d}".format(
    validation_metrics['coverage_error'], validation_metrics['average_num_of_labels'],
    validation_metrics['top_1'], validation_metrics['top_3'], validation_metrics['top_5'],
    validation_metrics['f1_micro'], validation_metrics['f1_macro'], validation_metrics['total_positive'])

# Dump the classifier and metrics
data_folder = os.path.join(svm_location, SVM_MODEL_NAME, data_type)
ensure_disk_location_exists(data_folder)
pickle.dump(clf, open(os.path.join(data_folder, CLASSIFIER_FILE.format(classifications_type)), "w"))
pickle.dump(training_metrics, open(os.path.join(data_folder, TRAINING_METRICS_FILENAME.format(classifications_type)), "w"))
pickle.dump(validation_metrics, open(os.path.join(data_folder, VALIDATION_METRICS_FILENAME.format(classifications_type)), "w"))

del X, y, Xv, yv


# Get the test data
info('Getting Test Data')
Xt = pickle.load(open(data_test_location, "r"))
test_data_docids = pickle.load(open(data_test_docids_location, "r"))
yt = get_label_data(classifications, test_data_docids, doc_classification_map)
# Test Metrics
info('Evaluating on Test Data')
ytp = clf.predict(Xt)
ytp_score = clf.decision_function(Xt)
print ytp
test_metrics = get_metrics(yt, ytp_score, ytp)
print "** Test Metrics: Cov Err: {:.3f}, Avg Labels: {:.3f}, \n\t\t Top 1: {:.3f}, Top 3: {:.3f}, Top 5: {:.3f}, \n\t\t F1 Micro: {:.3f}, F1 Macro: {:.3f}, Total Pos: {:,d}".format(
    test_metrics['coverage_error'], test_metrics['average_num_of_labels'],
    test_metrics['top_1'], test_metrics['top_3'], test_metrics['top_5'],
    test_metrics['f1_micro'], test_metrics['f1_macro'], test_metrics['total_positive'])
pickle.dump(test_metrics, open(os.path.join(data_folder, TEST_METRICS_FILENAME.format(classifications_type)), "w"))
