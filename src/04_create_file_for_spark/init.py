import os
import sys
from collections import namedtuple
import cPickle as pickle
import logging
from logging import info
sys.path.append(os.path.abspath('..'))
from utils.file import ensure_disk_location_exists


root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # adds a default StreamHanlder

root_location = "../../data/"
exports_location = root_location + "exported_data/"

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

preprocessed_location = os.path.join(root_location, "preprocessed_data", "separated_datasets")

ensure_disk_location_exists(preprocessed_location)

training_preprocessed_files_prefix = os.path.join(preprocessed_location, "training_docs_data_preprocessed-")
validation_preprocessed_files_prefix = os.path.join(preprocessed_location, "validation_docs_data_preprocessed-")
test_preprocessed_files_prefix = os.path.join(preprocessed_location, "test_docs_data_preprocessed-")

docs_only_preprocessed_file = os.path.join(root_location,"preprocessed_data", "docs_only_file.txt")


doc_classification_map = pickle.load(open(doc_classification_map_file))
sections = pickle.load(open(sections_file))
valid_classes = pickle.load(open(valid_classes_file))
valid_subclasses = pickle.load(open(valid_subclasses_file))
training_docs_list = pickle.load(open(training_docs_list_file))
validation_docs_list = pickle.load(open(validation_docs_list_file))
test_docs_list = pickle.load(open(test_docs_list_file))


class FullDocumentWriterBatchIterator(object):
    def __init__(self, filename_prefix, file_to_write_path):
        self.filename_prefix = filename_prefix
        self.file_to_write_path = file_to_write_path
        self.file_to_write = open(file_to_write_path, "a")
        self.curr_doc_index = 0
    def load_new_batch_in_memory(self):
        info("Loading new batch for index: {}".format(self.curr_doc_index))
        try:
            preproc_file  = open(self.filename_prefix + str(self.curr_doc_index), "r")
            info("Finished loading new batch of {} documents".format(self.curr_doc_index))
            return preproc_file
        except:
            return False
    def iterate_over_all_docs(self):
        preproc_file = self.load_new_batch_in_memory()
        while preproc_file is not False:
            for line in preproc_file:
                try:
                    doc_id, doc_content = line.split(" ", 1)
                    if is_real_doc(doc_id):
                        self.file_to_write.write(line + "\n")
                        self.file_to_write.write(line)
                        self.curr_doc_index += 1
                except:
                    continue
            preproc_file = self.load_new_batch_in_memory()
        self.file_to_write.close()


def is_real_doc(doc_id):
    return doc_id.find("_") == -1

# Training Documents
training_iterator = FullDocumentWriterBatchIterator(training_preprocessed_files_prefix,
                                                                      docs_only_preprocessed_file)
training_iterator.iterate_over_all_docs()

# Validation Documents
validation_iterator = FullDocumentWriterBatchIterator(validation_preprocessed_files_prefix,
                                                                    docs_only_preprocessed_file)
validation_iterator.iterate_over_all_docs()

# Test Documents
test_iterator = FullDocumentWriterBatchIterator(test_preprocessed_files_prefix,
                                                              docs_only_preprocessed_file)
test_iterator.iterate_over_all_docs()