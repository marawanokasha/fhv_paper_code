import os
import sys
import argparse
from collections import namedtuple
import cPickle as pickle
import numpy as np
import gzip
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

import logging
from logging import info

sys.path.append(os.path.abspath('..'))
from utils.classification import OneHotEncoder
from utils.file import ensure_disk_location_exists

root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # adds a default StreamHanlder

GLOBAL_VARS = namedtuple('GLOBAL_VARS', ['MODEL_NAME', 'DOC2VEC_MODEL_NAME', 'DOC2VEC_MODEL', 'DOC2VEC_RAW_MODEL_NAME'])

VOCAB_MODEL = "vocab_model"
MODEL_PREFIX = "model"
VALIDATION_MATRIX = "validation_matrix.pkl"
VALIDATION_DICT = "validation_dict.pkl"
TEST_MATRIX = "test_matrix.pkl"
TEST_DICT = "test_dict.pkl"
METRICS = "metrics.pkl"
CLASSIFIER = "classifier.pkl"
TYPE_CLASSIFIER= "{}_classifier.pkl"

TRAINING_DATA_MATRIX = "X_level_{}.npy"
TRAINING_LABELS_MATRIX = "y_{}.npy"
VALIDATION_DATA_MATRIX = "Xv_level_{}.npy"
VALIDATION_LABELS_MATRIX = "yv_{}.npy"
TEST_DATA_MATRIX = "Xt_level_{}.npy"
TEST_LABELS_MATRIX = "yt_{}.npy"
GZIP_EXTENSION = ".gz"


root_location = "../../data/"
exports_location = root_location + "exported_data/"
matrices_save_location = root_location + "fhv_matrices/"

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


parser = argparse.ArgumentParser(description='Create FHV matrices for the different levels')
parser.add_argument("-l", "--level", type=int, help="FHV level to create the matrix for")
args = parser.parse_args()

print args


LEVEL_TO_GENERATE = args.level


NUM_ABSTRACT_CHUNKS = 3
NUM_DESC_CHUNKS = 23
NUM_CLAIMS_CHUNKS = 4

LEVEL_1_ID = "{}"
LEVEL_2_ID = "{}_{}"
LEVEL_3_ID = "{}_{}_part-{}"

PART_LEVEL_NAME = "{}_{}"

DOCUMENT_ORDER = [
    (1, "document"),
    (2, "abstract"), (3, "abstract"),
    (2, "description"), (3, "description"),
    (2, "claims"), (3, "claims")
]
DOCUMENT_PART_SIZES = {

    "1_document": 1,
    "2_abstract": 1,
    "2_description": 1,
    "2_claims": 1,
    "3_abstract": NUM_ABSTRACT_CHUNKS,
    "3_description": NUM_DESC_CHUNKS,
    "3_claims": NUM_CLAIMS_CHUNKS
}

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
DOC2VEC_EPOCHS = 1 # we do our training manually one epoch at a time
DOC2VEC_MAX_EPOCHS = 8
REPORT_DELAY = 20 # report the progress every x seconds
REPORT_VOCAB_PROGRESS = 100000 # report vocab progress every x documents

DOC2VEC_MMAP = 'r'
DOC2VEC_EPOCH = 8

EMBEDDING_SIZE = DOC2VEC_SIZE
ZERO_VECTOR = [0] * DOC2VEC_SIZE

raw_model_name = 'doc2vec_size_{}_w_{}_type_{}_concat_{}_mean_{}_trainwords_{}_hs_{}_neg_{}_vocabsize_{}'.format(DOC2VEC_SIZE,
                        DOC2VEC_WINDOW,
                        'dm' if DOC2VEC_TYPE == 1 else 'pv-dbow',
                        DOC2VEC_CONCAT, DOC2VEC_MEAN,
                        DOC2VEC_TRAIN_WORDS,
                        DOC2VEC_HIERARCHICAL_SAMPLE,DOC2VEC_NEGATIVE_SAMPLE_SIZE,
                        str(DOC2VEC_MAX_VOCAB_SIZE)
                        )
raw_model_name = os.path.join(raw_model_name, "epoch_{}")
GLOBAL_VARS.DOC2VEC_RAW_MODEL_NAME = raw_model_name.format(DOC2VEC_EPOCH)


def get_part_ids(doc_id, part_level, part_name):
    """
    Returns the ids to look for, for a given document id, part level and part name
    ex get_part_ids(x, 3, "abstract) => ["x_abstract_part-1", "x_abstract_part-2", "x_abstract_part-3", ...]
    """
    if part_name == "document":
        return [LEVEL_1_ID.format(doc_id)]
    elif part_level == 2:
        return [LEVEL_2_ID.format(doc_id, part_name)]
    elif part_level == 3:
        ids = []
        for i in range(DOCUMENT_PART_SIZES[PART_LEVEL_NAME.format(part_level, part_name)]):
            ids.append(LEVEL_3_ID.format(doc_id, part_name, i+1))
        return ids


def get_sequence_insert_location(my_part_level, my_part_name, max_level):
    """
    for a given level and name, determines where its position in the sequence begins
    """
    assert DOCUMENT_PART_SIZES.get(PART_LEVEL_NAME.format(my_part_level, my_part_name)) is not None
    loc = 0
    for part_level, part_name in DOCUMENT_ORDER:
        if part_level <= max_level:
            if part_level == my_part_level and part_name == my_part_name:
                break
            else:
                loc += DOCUMENT_PART_SIZES[PART_LEVEL_NAME.format(part_level, part_name)]
    return loc


def create_labels(classifications, docs_list):
    one_hot_encoder = OneHotEncoder(classifications)
    classifications_set = set(classifications)
    labels_mat = np.zeros((len(docs_list), len(classifications)), dtype=np.int8)
    for i, doc_id in enumerate(docs_list):
        eligible_classifications = set(doc_classification_map[doc_id]) & classifications_set
        labels_mat[i][:] = one_hot_encoder.get_label_vector(eligible_classifications)
    return labels_mat


def fill_matrix(data_matrix, source_dict, docs_list, start_location, use_get=False):
    """
    the use_get flag is for doc2vec_model.docvecs since it doesnt support .get(), so we catch the exception and
    fill with zeros in that case. This should really happen very rarely (if ever) so this exception handling
    should not be a drain on performance
    """
    for i, doc_id in enumerate(docs_list):
        child_ids = get_part_ids(doc_id, part_level, part_name)

        j = start_location
        for child_id in child_ids:
            try:
                if not use_get or source_dict.get(child_id) is not None:
                    data_matrix[i][j] = source_dict[child_id]
                else:
                    info("ZERO_VECTOR for {}".format(child_id))
                    data_matrix[i][j] = ZERO_VECTOR
            except:
                info("ZERO_VECTOR for {}".format(child_id))
                data_matrix[i][j] = ZERO_VECTOR
            j += 1


sequence_size = sum([DOCUMENT_PART_SIZES["{}_{}".format(part_level, part_name)] for part_level, part_name in DOCUMENT_ORDER if part_level <= LEVEL_TO_GENERATE])
print sequence_size

X_data = np.ndarray((len(training_docs_list), sequence_size, EMBEDDING_SIZE), dtype=np.float32)
Xv_data = np.ndarray((len(validation_docs_list), sequence_size, EMBEDDING_SIZE), dtype=np.float32)
Xt_data = np.ndarray((len(test_docs_list), sequence_size, EMBEDDING_SIZE), dtype=np.float32)

info("********** Generating Matrices for LEVEL:{} ************".format(LEVEL_TO_GENERATE))

for part_level, part_name in DOCUMENT_ORDER:
    if part_level <= LEVEL_TO_GENERATE:

        info("======== Working on Level: {} => {}".format(part_level, part_name))

        sequence_insert_location = get_sequence_insert_location(part_level, part_name, LEVEL_TO_GENERATE)


        doc2vec_model_save_location = os.path.join(root_location,
                                                   "parameter_search_doc2vec_models_" + str(part_level) + '_' + part_name,
                                                   "full")

        placeholder_model_name = 'doc2vec_size_{}_w_{}_type_{}_concat_{}_mean_{}_trainwords_{}_hs_{}_neg_{}_vocabsize_{}_model_{}'.format(DOC2VEC_SIZE,
                                                                DOC2VEC_WINDOW,
                                                                'dm' if DOC2VEC_TYPE == 1 else 'pv-dbow',
                                                                DOC2VEC_CONCAT, DOC2VEC_MEAN,
                                                                DOC2VEC_TRAIN_WORDS,
                                                                DOC2VEC_HIERARCHICAL_SAMPLE,DOC2VEC_NEGATIVE_SAMPLE_SIZE,
                                                                str(DOC2VEC_MAX_VOCAB_SIZE),
                                                                str(part_level) + '_' + part_name
                                                                )
        GLOBAL_VARS.DOC2VEC_MODEL_NAME = placeholder_model_name
        placeholder_model_name = os.path.join(placeholder_model_name, "epoch_{}")
        epoch = DOC2VEC_EPOCH
        GLOBAL_VARS.MODEL_NAME = placeholder_model_name.format(epoch)


        info("Loading Doc2vec model: {}".format(GLOBAL_VARS.MODEL_NAME))
        doc2vec_model = Doc2Vec.load(os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME, MODEL_PREFIX), mmap=DOC2VEC_MMAP)
        info("Loading Validation Dict")
        validation_dict = dict(pickle.load(gzip.open(os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME, VALIDATION_DICT + GZIP_EXTENSION))))
        info("Loading Test Dict")
        test_dict = dict(pickle.load(gzip.open(os.path.join(doc2vec_model_save_location, GLOBAL_VARS.MODEL_NAME, TEST_DICT + GZIP_EXTENSION))))

        part_level_name = PART_LEVEL_NAME.format(part_level, part_name)

        info("Filling training matrix")
        fill_matrix(X_data, doc2vec_model.docvecs, training_docs_list, sequence_insert_location, use_get=False)
        info("Filling validation matrix")
        fill_matrix(Xv_data, validation_dict, validation_docs_list, sequence_insert_location, use_get=True)
        info("Filling test matrix")
        fill_matrix(Xt_data, test_dict, test_docs_list, sequence_insert_location, use_get=True)

ensure_disk_location_exists(os.path.join(matrices_save_location, GLOBAL_VARS.DOC2VEC_RAW_MODEL_NAME))
info("Saving training matrix")
np.save(open(os.path.join(matrices_save_location, GLOBAL_VARS.DOC2VEC_RAW_MODEL_NAME,
                          TRAINING_DATA_MATRIX.format(LEVEL_TO_GENERATE)), "w"), X_data)
info("Saving validation matrix")
np.save(open(os.path.join(matrices_save_location, GLOBAL_VARS.DOC2VEC_RAW_MODEL_NAME,
                          VALIDATION_DATA_MATRIX.format(LEVEL_TO_GENERATE)), "w"), Xv_data)
info("Saving test matrix")
np.save(open(os.path.join(matrices_save_location, GLOBAL_VARS.DOC2VEC_RAW_MODEL_NAME,
                          TEST_DATA_MATRIX.format(LEVEL_TO_GENERATE)), "w"), Xt_data)


## Labels

classifications_to_create = [
    ("sections", sections),
    ("classes", valid_classes),
    ("subclasses", valid_subclasses)
]

for classifications_type, classifications in classifications_to_create:
    info("Creating Training Labels for {}".format(classifications_type))
    y = create_labels(classifications, training_docs_list)
    info("Creating Validation Labels for {}".format(classifications_type))
    yv = create_labels(classifications, validation_docs_list)
    info("Creating Test Labels for {}".format(classifications_type))
    yt = create_labels(classifications, test_docs_list)

    ensure_disk_location_exists(os.path.join(matrices_save_location, GLOBAL_VARS.DOC2VEC_RAW_MODEL_NAME))
    np.save(open(os.path.join(matrices_save_location, GLOBAL_VARS.DOC2VEC_RAW_MODEL_NAME,
                                  TRAINING_LABELS_MATRIX.format(classifications_type)), "w"), y)
    np.save(open(os.path.join(matrices_save_location, GLOBAL_VARS.DOC2VEC_RAW_MODEL_NAME,
                                  VALIDATION_LABELS_MATRIX.format(classifications_type)), "w"), yv)
    np.save(open(os.path.join(matrices_save_location, GLOBAL_VARS.DOC2VEC_RAW_MODEL_NAME,
                                  TEST_LABELS_MATRIX.format(classifications_type)), "w"), yt)