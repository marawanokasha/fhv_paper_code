import cPickle as pickle
import os
import urllib2
import time
import json

import logging
from logging import info

from multiprocessing import Pool as ThreadPool

from text import get_sentences, sentence_wordtokenizer
from parse_utils import *


root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # adds a default StreamHanlder


root_location = "../../data/"
exports_location = root_location + "exported_data/"
doc_classifications_map_file = exports_location + "doc_classification_map.pkl"
training_docs_list_file = exports_location + "training_docs_list.pkl"
validation_docs_list_file = exports_location + "validation_docs_list.pkl"
test_docs_list_file = exports_location + "test_docs_list.pkl"

doc_classification_map = pickle.load(open(doc_classifications_map_file))
training_docs_list = pickle.load(open(training_docs_list_file))
validation_docs_list = pickle.load(open(validation_docs_list_file))
test_docs_list = pickle.load(open(test_docs_list_file))


ABSTRACT_ID = "{}_abstract"
DESC_ID = "{}_description"
CLAIMS_ID = "{}_claims"

ABSTRACT_PART_ID = "{}_abstract_part-{}"
DESC_PART_ID = "{}_description_part-{}"
CLAIMS_PART_ID = "{}_claims_part-{}"

BATCH_SIZE = 10000

preprocessed_location = root_location + "preprocessed_data/extended_pv_abs_desc_claims_full_chunks/"
TRAINING_PREPROCESSED_FILES_PREFIX = preprocessed_location + "extended_pv_training_docs_data_preprocessed-"
VALIDATION_PREPROCESSED_FILES_PREFIX = preprocessed_location + "extended_pv_validation_docs_data_preprocessed-"
TEST_PREPROCESSED_FILES_PREFIX = preprocessed_location + "extended_pv_test_docs_data_preprocessed-"

if not os.path.exists(preprocessed_location):
    os.makedirs(preprocessed_location)


NUM_ABSTRACT_PARTS = 3
NUM_DESC_PARTS = 23
NUM_CLAIMS_PARTS = 4


ES_URL = 'http://localhost:9200/patents/patent/{}'


def get_patent(doc_id):
    url_to_fetch = ES_URL.format(doc_id)

    response = urllib2.urlopen(url_to_fetch, timeout=60)
    patent_content = response.read()

    patent_object = json.loads(patent_content)['_source']
    return patent_object



def multithreaded_extended_batch_creation(start_index):

    if os.path.exists(FILE_PREFIX + str(start_index)):
        info("Batch {} already exists, skipping..".format(start_index))
        return

    info("Batch creation working on {}\n".format(start_index))
    token_lines = []
    start_time = time.time()

    for doc_index, doc_id in enumerate(DOCS_LIST[start_index:]):

        patent_doc = get_patent(doc_id)

        # Abstract
        abstract = patent_doc['abstract'][0]
        root = ET.fromstring(abstract.encode('utf-8'))
        abs_paragraphs = get_adjusted_paragraphs(root)

        # Description
        desc = patent_doc['description'][0]
        root = ET.fromstring(desc.encode('utf-8'))
        desc_paragraphs = get_adjusted_paragraphs(root)

        # Claims
        claims = patent_doc['claims'][0]
        root = ET.fromstring(claims.encode('utf-8'))
        claims_paragraphs = get_adjusted_paragraphs(root, conc_sentences=False)


        abstract_tokens = sum([sentence_wordtokenizer(parag) for parag in abs_paragraphs], [])
        desc_tokens = sum([sentence_wordtokenizer(parag) for parag in desc_paragraphs], [])
        claims_tokens = sum([sentence_wordtokenizer(parag) for parag in claims_paragraphs], [])


        # lists of list of tokens
        doc_tokens_list = [doc_id]  + abstract_tokens + desc_tokens + claims_tokens
        abstract_tokens_list = [ABSTRACT_ID.format(doc_id)] + abstract_tokens
        description_tokens_list = [DESC_ID.format(doc_id)] + desc_tokens
        claims_tokens_list = [CLAIMS_ID.format(doc_id)] + claims_tokens

        # now add the tokens lists that will be written to the file
        token_lines.append(doc_tokens_list)
        token_lines.append(abstract_tokens_list)
        token_lines.append(description_tokens_list)
        token_lines.append(claims_tokens_list)

        for i in range(NUM_ABSTRACT_PARTS):
            start, end = get_doc_range(i, len(abstract_tokens), NUM_ABSTRACT_PARTS)
            token_lines.append([ABSTRACT_PART_ID.format(doc_id, i+1)] + abstract_tokens[start: end])

        for i in range(NUM_DESC_PARTS):
            start, end = get_doc_range(i, len(desc_tokens), NUM_DESC_PARTS)
            token_lines.append([DESC_PART_ID.format(doc_id, i+1)] + desc_tokens[start: end])

        for i in range(NUM_CLAIMS_PARTS):
            start, end = get_doc_range(i, len(claims_tokens), NUM_CLAIMS_PARTS)
            token_lines.append([CLAIMS_PART_ID.format(doc_id, i+1)] + claims_tokens[start: end])

        if doc_index % 1000 == 0: info("Doc: {:6} -> Total Lines to write: {:8}".format(start_index + doc_index, len(token_lines)))
        if doc_index >= BATCH_SIZE - 1:
            break

    duration = time.time() - start_time
    info("Finished batch {} of size {:d} in {:.0f}m {:.0f}s".format(start_index, BATCH_SIZE, * divmod(duration, 60)))
    info("For index {}, the actual number of lines written is: {}".format(start_index, len(token_lines)))

    write_batch(FILE_PREFIX, token_lines, start_index)
    del token_lines


def get_doc_range(i, number_of_tokens, number_of_parts):
    start, end = 0,0
    if number_of_tokens < number_of_parts:
        if i==0:
            return 0, None
        else:
            return number_of_tokens,None
    if i == 0:
        start = 0
    else:
        start = (number_of_tokens / number_of_parts) * i
    if i+1 == number_of_parts:
        end = None
    else:
        end = (number_of_tokens / number_of_parts) * (i+1)
    return start, end


def write_batch(file_prefix, batch_lines, batch_start):
    if len(batch_lines):
        print "writing batch %d" % batch_start
        with open(file_prefix + str(batch_start), 'w') as batch_file:
            for line in batch_lines:
                batch_file.write((u" ".join(line) + "\n").encode('utf-8'))
        print "finished writing batch %d" % batch_start



# depending on how much memory you have, set this accordingly, as every thread takes a considerable amount of memory
NUM_THREADS_TO_USE = 8

# ## Training


DOCS_LIST = training_docs_list
FILE_PREFIX = TRAINING_PREPROCESSED_FILES_PREFIX
SAMPLE_SIZE = len(training_docs_list)
batches = range(0, (divmod(SAMPLE_SIZE, BATCH_SIZE)[0]+1) * BATCH_SIZE, BATCH_SIZE)

try:
    pool = ThreadPool(NUM_THREADS_TO_USE)
    batches = range(0, (divmod(SAMPLE_SIZE, BATCH_SIZE)[0]+1) * BATCH_SIZE, BATCH_SIZE )
    indices = pool.map(multithreaded_extended_batch_creation, batches)
    pool.close()
    pool.terminate()
finally:
    pool.close()
    pool.terminate()


# ## Validation


DOCS_LIST = validation_docs_list
FILE_PREFIX = VALIDATION_PREPROCESSED_FILES_PREFIX
SAMPLE_SIZE = len(validation_docs_list)

try:
    pool = ThreadPool(NUM_THREADS_TO_USE)
    batches = range(0, (divmod(SAMPLE_SIZE, BATCH_SIZE)[0]+1) * BATCH_SIZE, BATCH_SIZE )
    indices = pool.map(multithreaded_extended_batch_creation, batches)
    pool.close()
    pool.terminate()
finally:
    pool.close()
    pool.terminate()


# ## Test

DOCS_LIST = test_docs_list
FILE_PREFIX = TEST_PREPROCESSED_FILES_PREFIX
SAMPLE_SIZE = len(test_docs_list)

try:
    pool = ThreadPool(NUM_THREADS_TO_USE)
    # +1 since range is end-exclusive
    batches = range(0, (divmod(SAMPLE_SIZE, BATCH_SIZE)[0]+1) * BATCH_SIZE, BATCH_SIZE )
    indices = pool.map(multithreaded_extended_batch_creation, batches)
    pool.close()
    pool.terminate()
finally:
    pool.close()
    pool.terminate()



