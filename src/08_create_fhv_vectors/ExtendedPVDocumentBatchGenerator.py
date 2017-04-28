import logging
from logging import info
import sys
import gzip
import io
import numpy as np
from multiprocessing import Process, Queue
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from functools import partial
root = logging.getLogger()
for handler in root.handlers[:]:
    root.removeHandler(handler)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) # adds a default StreamHanlder


class ExtendedPVDocumentBatchGenerator(Process):
    def __init__(self, filename_prefix, queue, start_file=0, offset=10000):
        super(ExtendedPVDocumentBatchGenerator, self).__init__()
        self.queue = queue
        self.offset = offset
        self.filename_prefix = filename_prefix
        self.files_loaded = start_file - offset

    def run(self):
        cur_file = None
        while True:
            try:
                if cur_file is None:
                    info("Loading new file for index: {}".format(str(self.files_loaded + self.offset)))
                    cur_file = io.BufferedReader(gzip.open(self.filename_prefix + str(self.files_loaded + self.offset) + '.gz'))
                    self.files_loaded += self.offset
                for line in cur_file:
                    self.queue.put(line)
                cur_file.close()
                cur_file = None
            except IOError:
                self.queue.put(False, block=True, timeout=None)
                info("All files are loaded - last file: {}".format(str(self.files_loaded + self.offset)))
                sys.exit()


class BatchWrapper(object):
    def __init__(self, training_preprocessed_files_prefix, buffer_size=10000, batch_size=10000, level=1, level_type=None):
        assert batch_size <= 10000 or batch_size is None
        self.level = level
        self.level_type = level_type[0]
        self.batch_size = batch_size
        self.q = Queue(maxsize=buffer_size)
        self.p = ExtendedPVDocumentBatchGenerator(training_preprocessed_files_prefix, queue=self.q,
                                                  start_file=0, offset=10000)
        self.p.start()
        self.cur_data = []

    def is_correct_type(self, doc_id):
        parts = doc_id.split("_")
        len_parts = len(parts)
        if len_parts == self.level:
            if len_parts == 1:
                return True
            if len_parts == self.level and (parts[1][0] == self.level_type or self.level_type is None):
                return True
        return False

    def return_sentences(self, line):
        line_array = tuple(line.split(" "))
        doc_id = line_array[0]
        if not self.is_correct_type(doc_id):
            return False
        line_array = line_array[1:]
        len_line_array = len(line_array)
        curr_batch_iter = 0
        # divide the document to batches according to the batch size
        sentences = []
        if self.batch_size is None:
            sentences.append((doc_id, line_array))
        else:
            while curr_batch_iter < len_line_array:
                sentences.append(LabeledSentence(words=line_array[curr_batch_iter: curr_batch_iter + self.batch_size], tags=[doc_id]))
                curr_batch_iter += self.batch_size
        return tuple(sentences)

    def __iter__(self):
        while True:
            item = self.q.get(block=True)
            if item is False:
                self.p.terminate()
                raise StopIteration()
            else:
                try:
                    sentences = self.return_sentences(item)
                except:
                    print(item)
                    raise StopIteration()
                if not sentences:
                    None
                else:
                    for sentence in sentences:
                        yield sentence
