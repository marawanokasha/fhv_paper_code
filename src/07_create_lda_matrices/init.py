import os
import sys
import random
import cPickle as pickle
import argparse
import gzip
from sklearn.decomposition import LatentDirichletAllocation

import logging
from logging import info

sys.path.append(os.path.abspath('..'))
from utils.file import ensure_disk_location_exists


root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # adds a default StreamHanlder

RANDOM_SEED = 10000
random.seed(RANDOM_SEED)
SVM_SEED = 1234
LDA_SEED = 1234
NUM_CORES = 24

MODEL_FILE = 'lda_model.pkl'
VALIDATION_METRICS_FILENAME= '{}_validation_metrics.pkl'
TRAINING_METRICS_FILENAME = '{}_training_metrics.pkl'
TEST_METRICS_FILENAME = '{}_test_metrics.pkl'
GZIP_EXTENSION = '.gz'


root_location = "../../data/"
exports_location = root_location + "exported_data/"
lda_location = root_location + "extended_pv_lda/"

classifications_index_file = os.path.join(exports_location, "classifications_index.pkl")
doc_classification_map_file = os.path.join(exports_location, "doc_classification_map.pkl")
sections_file = os.path.join(exports_location, "sections.pkl")
classes_file = os.path.join(exports_location, "classes.pkl")
subclasses_file = os.path.join(exports_location, "subclasses.pkl")
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
classes = pickle.load(open(classes_file))
subclasses = pickle.load(open(subclasses_file))
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

parser = argparse.ArgumentParser(description='Run LDA on BOW data and create the training, validation and test matrices')
parser.add_argument("-d", "--dataType", choices=possible_data_types, required=True)
args = parser.parse_args()

print args

data_type = args.dataType


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

info("=============== Creating LDA Model of data: {} ================".format(data_type))

data_training_location = os.path.join(exports_location, "{}_training_sparse_data.pkl".format(data_type))
data_training_docids_location = os.path.join(exports_location, "{}_training_sparse_docids.pkl".format(data_type))
data_validation_location = os.path.join(exports_location, "{}_validation_sparse_data.pkl".format(data_type))
data_validation_docids_location = os.path.join(exports_location, "{}_validation_sparse_docids.pkl".format(data_type))
data_test_location = os.path.join(exports_location, "{}_test_sparse_data.pkl".format(data_type))
data_test_docids_location = os.path.join(exports_location, "{}_test_sparse_docids.pkl".format(data_type))

# Get the training data
info('Getting Training Data')
X = pickle.load(open(data_training_location, "r"))
info('Doing LDA decomposition')
lda = LatentDirichletAllocation(n_topics=LDA_TOPICS, max_iter=LDA_ITERATIONS, learning_method=LDA_LEARNING_METHOD, \
                               learning_decay=LDA_DECAY, batch_size=LDA_BATCH_SIZE, \
                                evaluate_every=LDA_EVALUATE_EVERY, n_jobs=NUM_CORES, verbose=LDA_VERBOSE, random_state=LDA_SEED)
lda.fit(X)

# Dump the LDA model
data_folder = os.path.join(lda_location, LDA_MODEL_NAME, data_type)
ensure_disk_location_exists(data_folder)
pickle.dump(lda, gzip.open(os.path.join(data_folder, MODEL_FILE + GZIP_EXTENSION), "w"))


## Creating the training, validation, test matrices


data_folder = os.path.join(lda_location, LDA_MODEL_NAME, data_type)


### Training

lda_data_training_location = os.path.join(data_folder, "lda_training_data.pkl")
X_lda_training = lda.transform(X)
pickle.dump(X_lda_training, open(lda_data_training_location, "w"))


### Validation

Xv = pickle.load(open(data_validation_location,'r'))
lda_data_validation_location = os.path.join(data_folder, "lda_validation_data.pkl")
X_lda_validation = lda.transform(Xv)
pickle.dump(X_lda_validation, open(lda_data_validation_location, "w"))


### Test

Xt = pickle.load(open(data_test_location, "r"))
lda_data_test_location = os.path.join(data_folder, "lda_test_data.pkl")
X_lda_test = lda.transform(Xt)
pickle.dump(X_lda_test, open(lda_data_test_location, "w"))