from pyspark import SparkContext, SparkConf
import cPickle as pickle
import json
import os
import random

es_server = "localhost"
es_port = "9200"

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


TEST_SET_PERCENTAGE = 0.2
VALIDATION_IN_TRAINING_PERCENTAGE = 0.2
MIN_DOCUMENTS_FOR_TEST = 1
MIN_DOCUMENTS_FOR_VALIDATION = 1

# minimum number of documents per classification to consider it valid for being used in classification
INVALID_CLASSIFICATION_LIMIT = 3


conf = SparkConf().setAppName('Getting utility Data').setMaster("local")
sc = SparkContext()

read_conf = {
    'es.nodes': es_server,
    'es.port': es_port,
    'es.resource': 'patents/patent',
    'es.query': '{ "query" : { "match_all" : {} }}',
    'es.scroll.keepalive': '120m',
    'es.scroll.size': '1000',
    'es.http.timeout': '20m'
}
data = sc.newAPIHadoopRDD(
    inputFormatClass = 'org.elasticsearch.hadoop.mr.EsInputFormat',
    keyClass = 'org.apache.hadoop.io.NullWritable',
    valueClass = 'org.elasticsearch.hadoop.mr.LinkedMapWritable',
    conf = read_conf
)


def compare_classifications(x,y):
    len_comp = cmp(len(x), len(y))
    if len_comp == 0:
        return cmp(x,y)
    return len_comp


def get_classes(ipc_classification):
    sections = []
    classes = []
    subclasses = []
    for classification in ipc_classification:
        # we do the check because some documents have repetitions
        section_name = classification['section']
        class_name = classification['section'] + "-" + classification['class']
        subclass_name = classification['section'] + "-" + classification['class'] + "-" + classification['subclass']
        if section_name not in sections:
            sections.append(section_name)
        if class_name not in classes:
            classes.append(class_name)
        if subclass_name not in subclasses:
            subclasses.append(subclass_name)
    return {"sections": sections, "classes": classes, "subclasses": subclasses}


doc_objs = data.map(lambda x: json.loads(x))
doc_class_map = doc_objs.map(lambda (doc_id, doc): (doc_id, get_classes(doc['classification-ipc']))).cache()
doc_classification_map = doc_class_map.map(lambda (doc_id, classification_obj): (doc_id, sorted(reduce(lambda x, lst: x + lst, classification_obj.values(), [])))).collectAsMap()
doc_count = len(doc_classification_map)
# contains [(classification,  list of docs)]
# second list comprehension is to get list of lists [["A", "B"],["A-01","B-03"]] to one list ["A", "B", "A-01","B-03"], we could have also used a reduce as in doc_classifications_map
classifications_index = doc_class_map.flatMap(lambda (doc_id, classifications_obj): [(classification, doc_id) for classification in [classif for cat in classifications_obj.values() for classif in cat]])\
    .groupByKey().map(lambda (classf, classf_docs): (classf, list(set(classf_docs)))).collectAsMap()
sections = sorted(doc_class_map.flatMap(lambda (doc_id, classifications): classifications['sections']).distinct().collect())
classes = sorted(doc_class_map.flatMap(lambda (doc_id, classifications): classifications['classes']).distinct().collect())
subclasses = sorted(doc_class_map.flatMap(lambda (doc_id, classifications): classifications['subclasses']).distinct().collect())
classifications = sorted(classifications_index.keys(), cmp=compare_classifications)


invalid_classes = set()
invalid_subclasses = set()
for clsf in classifications_index.akeys():
    if len(classifications_index[clsf]) < INVALID_CLASSIFICATION_LIMIT:
        if clsf in classes:
            invalid_classes.add(clsf)
        if clsf in subclasses:
            invalid_subclasses.add(clsf)
valid_classes = sorted(list(set(classes) - invalid_classes))
valid_subclasses = sorted(list(set(subclasses) - invalid_subclasses))


training_documents = set()
validation_documents = set()
test_documents = set()
for (classf, documents) in classifications_index.items():
    # only worry about subclasses, classes and sections will be already included
    if(classf in sections or classf in classes): pass

    # remove any documents that have already been picked before
    docs_set = set(documents)
    docs_set-=training_documents
    docs_set-=validation_documents
    docs_set-=test_documents

    base_test_docs_num = int(len(docs_set)* TEST_SET_PERCENTAGE)
    num_test_docs = base_test_docs_num if base_test_docs_num > 0 else MIN_DOCUMENTS_FOR_TEST if MIN_DOCUMENTS_FOR_TEST < len(docs_set) else 0
    print len(docs_set), num_test_docs
    classif_test_docs = random.sample(docs_set, num_test_docs)

    remaining_docs = docs_set.difference(set(classif_test_docs))
    base_validation_docs_num = int(len(remaining_docs)* VALIDATION_IN_TRAINING_PERCENTAGE)
    num_validation_docs = base_validation_docs_num if base_validation_docs_num > 0 else MIN_DOCUMENTS_FOR_VALIDATION if MIN_DOCUMENTS_FOR_VALIDATION < len(remaining_docs) else 0
    classif_validation_docs = random.sample(remaining_docs, num_validation_docs)

    classif_training_docs = set(remaining_docs).difference(set(classif_validation_docs))

    training_documents.update(classif_training_docs)
    validation_documents.update(classif_validation_docs)
    test_documents.update(classif_test_docs)


# Save to pickle files to be used in later steps
pickle.dump(sections, open(sections_file, 'w'))
pickle.dump(classes, open(classes_file, 'w'))
pickle.dump(subclasses, open(subclasses_file, 'w'))
pickle.dump(valid_classes, open(valid_classes_file, "w"))
pickle.dump(valid_subclasses, open(valid_subclasses_file, "w"))
pickle.dump(classifications, open(classifications_file, 'w'))
pickle.dump(classifications_index, open(classifications_index_file, 'w'))
pickle.dump(doc_classification_map, open(doc_classification_map_file, 'w'))

pickle.dump(training_documents, open(training_docs_list_file, 'w'))
pickle.dump(validation_documents, open(validation_docs_list_file, 'w'))
pickle.dump(test_documents, open(test_docs_list_file, 'w'))