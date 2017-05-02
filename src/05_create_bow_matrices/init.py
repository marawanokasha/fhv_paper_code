from pyspark import SparkContext, SparkConf
import os
import sys
import random
import cPickle as pickle
import logging
import sklearn.feature_extraction

from postings_util import merge_postings, create_doc_index, get_chi_index, get_term_dictionary, \
    calculate_bm25, calculate_sublinear_tf, calculate_sublinear_tf_idf, calculate_tf_idf

sys.path.append(os.path.abspath('..'))
from utils.text import simple_tokenizer
from utils.file import ensure_disk_location_exists

root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # adds a default StreamHanlder


conf = SparkConf().setAppName('Getting utility Data').setMaster("local")
sc = SparkContext()


MIN_DOCUMENTS = 5
TOP_N_FEATURES = 10000


root_location = "../../data/"
exports_location = root_location + "exported_data/"

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

docs_only_preprocessed_file = os.path.join(root_location,"preprocessed_data", "docs_only_file.txt")
bow_data_location = os.path.join(root_location, "bow_data")

ensure_disk_location_exists(bow_data_location)

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
classifications_index = pickle.load(open(classifications_index_file))

doc_count = len(doc_classification_map)
classifications_index_set = {k:set(docs) for k,docs in classifications_index.iteritems()} # accelerates the chi squared calculation a lot
training_docs_set = set(training_docs_list)
validation_docs_set = set(validation_docs_list)
test_docs_set = set(test_docs_list)


## Create Term Postings Lists

doc_text_objs = sc.textFile(docs_only_preprocessed_file).map(lambda line: line.split(" ", 1))
# Create Postings List
postings_lists = doc_text_objs.flatMap(lambda (doc_id, doc): simple_tokenizer(doc, doc_id)).reduceByKey(lambda x,y: merge_postings(x,y))
# the second condition is made specifically for num_indic, as it usually is in all docs and that messes up chi
min_doc_postings_lists = postings_lists.filter(lambda (x,y): len(y) > MIN_DOCUMENTS and len(y) < doc_count)
doc_lengths_dict = doc_text_objs.map(lambda (doc_id, document_text): (doc_id, len(document_text))).collectAsMap()
avg_doc_length = sum(doc_lengths_dict.values())/len(doc_lengths_dict)


## Caclculate Top features using Chi squared scoring

term_accepted_chi_list_with_scores = get_chi_index(min_doc_postings_lists, classifications_index_set, subclasses, doc_count).takeOrdered(TOP_N_FEATURES, lambda (term,score): -score)
term_accepted_chi_list = map(lambda (x,y): x, term_accepted_chi_list_with_scores)
# gets a bit slower at the end but finishes eventually
term_dictionary = get_term_dictionary(term_accepted_chi_list)
min_doc_postings_lists = min_doc_postings_lists.filter(lambda (term, postings): term in term_accepted_chi_list).cache()
number_of_terms = min_doc_postings_lists.count()
term_df_map = min_doc_postings_lists.map(lambda (term, postings): (term, len(postings))).collectAsMap()


## Create Training, validation and test sets using the top Chi squared features

training_min_doc_postings_lists = min_doc_postings_lists.map(lambda (term, postings): (term, {doc_id:postings[doc_id] for doc_id in postings if doc_id in training_docs_set}))
validation_min_doc_postings_lists = min_doc_postings_lists.map(lambda (term, postings): (term, {doc_id:postings[doc_id] for doc_id in postings if doc_id in validation_docs_set}))
test_min_doc_postings_lists = min_doc_postings_lists.map(lambda (term, postings): (term, {doc_id:postings[doc_id] for doc_id in postings if doc_id in test_docs_set}))


## Now create the data matrices to be used for classification

data_type_postings = [
    ("training", training_min_doc_postings_lists),
    ("validation", validation_min_doc_postings_lists),
    ("test", test_min_doc_postings_lists)
]

for data_type, doc_postings_list in data_type_postings:
    tf_postings = doc_postings_list
    tf_doc_index_training = create_doc_index(tf_postings, "tf")

    sublinear_tf_postings = tf_postings.mapValues(lambda postings: {docId:  calculate_sublinear_tf(tf) for docId, tf in postings.items()})
    sublinear_tf_doc_index_training = create_doc_index(sublinear_tf_postings, "tf-sublinear")

    tf_idf_postings = tf_postings.mapValues(lambda postings: {docId:  calculate_tf_idf(tf, len(postings), len(training_docs_list)) for docId, tf in postings.items()})
    tf_idf_doc_index_training = create_doc_index(tf_idf_postings, "tf-idf")

    sublinear_tf_idf_postings = tf_postings.mapValues(lambda postings: {docId:  calculate_sublinear_tf_idf(tf, len(postings), len(training_docs_list)) for docId, tf in postings.items()})
    sublinear_tf_idf_doc_index_training = create_doc_index(sublinear_tf_idf_postings, "sublinear-tf-idf")

    bm25_postings = tf_postings.mapValues(lambda postings: {docId: calculate_bm25(tf, len(postings), len(training_docs_list), doc_lengths_dict[docId], avg_doc_length) for docId, tf in postings.items()})
    bm25_doc_index_training = create_doc_index(bm25_postings, "bm25")

    all_bow_types = [
        ("bm25", bm25_doc_index_training),
        ("sublinear_tf", sublinear_tf_doc_index_training),
        ("tf_idf", tf_idf_doc_index_training),
        ("tf", tf_doc_index_training),
        ("sublinear_tf_idf", sublinear_tf_idf_doc_index_training)
    ]
    for (name, doc_index_rdd) in all_bow_types:
        v = sklearn.feature_extraction.DictVectorizer(sparse=True, dtype=float)
        doc_index = doc_index_rdd.collect()
        list_dicts = [y[1] for y in doc_index]
        doc_ids = [y[0] for y in doc_index]
        X = v.fit_transform(list_dicts)
        pickle.dump(X, open(os.path.join(bow_data_location, "{}_{}_sparse_data.pkl".format(name, data_type)),"w"))
        pickle.dump(doc_ids, open(os.path.join(bow_data_location, "{}_{}_sparse_docids.pkl".format(name, data_type)), "w"))
        del doc_index
        del X
        print name
