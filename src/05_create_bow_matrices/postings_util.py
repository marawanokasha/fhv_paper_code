import math
import json


BM25_K = 1.5  # controls power of tf component
BM25_b = 0.75  # controls the BM25 length normalization


def merge_postings(postings_list1, postings_list2):
    # key could be either a doc id or a term
    for key in postings_list2:
        if postings_list1.get(key):
            postings_list1[key] += postings_list2[key]
        else:
            postings_list1[key] = postings_list2[key]
    return postings_list1

def get_term_dictionary(terms):
    """
    Maps string terms to indexes in an array
    """
    term_dictionary = {}
    term_array = [None] * len(terms)
    def put(key):
        hashvalue = hashfunction(key, len(term_array))
        if term_array[hashvalue] == None:
            term_array[hashvalue] = key
            return hashvalue
        else:
            nextslot = rehash(hashvalue, len(term_array))
            while term_array[nextslot] != None:
                nextslot = rehash(nextslot, len(term_array))
            if term_array[nextslot] == None:
                term_array[nextslot] = key
                return nextslot
    def hashfunction(key, size):
        return hash(key) % size
    def rehash(oldhash, size):
        return (oldhash + 1) % size
    i = 0
    for term in terms:
        corresponding_index = put(term)
        term_dictionary[term] = corresponding_index
        i+=1
        if i%10000 == 0: print "finished " + str(i)
    return term_dictionary


def get_doc_index(term, postings_list, term_dictionary):
    return [(doc_id, {term_dictionary[term]: postings_list[doc_id]}) for doc_id in postings_list]


def calculate_sublinear_tf(tf):
    # laplace smoothing with +1 in case of term with no documents (useful during testing)
    return math.log10(1 + tf)


def calculate_tf_idf(tf, df, N):
    # laplace smoothing with +1 in case of term with no documents (useful during testing)
    return tf * math.log10((N+1) / (df + 1))


def calculate_sublinear_tf_idf(tf, df, N):
    # laplace smoothing with +1 in case of term with no documents (useful during testing)
    return calculate_sublinear_tf(tf) * math.log10((N+1) / (df + 1))


def calculate_bm25(tf, df, N, d_len, d_avg):
    idf = max(0, math.log10((N-df + 0.5)/(df+0.5))) # in rare cases where the df is over 50% of N, this could become -ve, so we guard against that
    tf_comp = float(((BM25_K + 1) * tf)) / ( BM25_K * ((1-BM25_b) + BM25_b*(float(d_len)/d_avg)) + tf)
    return tf_comp * idf


def create_doc_index(term_index, term_dictionary):
    return term_index \
        .flatMap(lambda (term, postings_list): get_doc_index(term, postings_list, term_dictionary)) \
        .reduceByKey(lambda x, y: merge_postings(x, y))


def get_chi_index(term_index, classifications_index_set, subclasses, number_of_docs):
    return term_index.map(lambda (term, postings_list): (term, calculate_chi_squared(postings_list.keys(), classifications_index_set, subclasses, number_of_docs)))


def calculate_chi_squared(document_list, classifications_index_set, subclasses, number_of_docs):
    """
    Chi squared is the ratio of the difference between actual frequency and expected frequency of a term relative to the expected frequency
    summed up across all classes and whether the term appears or not
    Here we calculate the average chi squared score which is one of two options in multi-label classification (the other being max)
    """
    chi_score = 0
    N = len(document_list)
    doc_set = set(document_list)
    Nt1 = N # actual collection frequency of having the word
    Nt0 = number_of_docs - N # actual collection frequency of not having the word
    Pt1 = float(N)/ number_of_docs # probability of the term happening
    Pt0 = float(number_of_docs - N)/ number_of_docs # probablility of the term not happening
    #print "Docs Stats: Term present in %d (%.7f), Not Present in %d (%.7f) " % (Nt1, Pt1, Nt0, Pt0)
    for subclass in subclasses:
        # this condition is only required for when using a sample because some subclasses may not have docs
        if len(classifications_index_set[subclass]) > 0:
            Pc1 = float(len(classifications_index_set[subclass]))/ number_of_docs # probability of the class happening
            Pc0 = 1 - Pc1
            Pt1c1 = float(len(doc_set & classifications_index_set[subclass])) / number_of_docs
            Pt1c0 = Pt1 - Pt1c1
            Pt0c1 = Pc1 - Pt1c1
            Pt0c0 = 1 - Pt1c0 - Pt0c1 - Pt1c1

            cat_chi_score = (number_of_docs * math.pow(Pt1c1 * Pt0c0 - Pt1c0 * Pt0c1, 2))/(Pt1 * Pt0 * Pc1 * Pc0)
            # calculate average chi score
            chi_score += Pc1 * cat_chi_score
            #print "subclass %s: %.7f, %.7f, %.7f, %.7f, %.7f, %.7f" % (subclass, Pc1, Pt1c1, Pt1c0, Pt0c1, Pt0c0, chi_score)
    return chi_score
