# -*- coding: utf-8 -*-

import string
import nltk
from nltk.tokenize import RegexpTokenizer
import re

NUMBER_INDICATOR = "num_indic"
CURRENCY_INDICATOR = "curr_indic"
CHEMICAL_INDICATOR = "chem_indic"
MIN_SIZE = 1

MIN_SENT_LENGTH = 50
MAX_NON_TABLE_ROW_LENGTH = 100

extra_abbrv = [u'u.s', u'fig', u'figs', u'no', u'ser',
               u'jan', u'feb', u'mar', u'apr', u'may', u'jun', u'jul', u'aug', u'sep', u'oct', u'nov', u'dec',
               u'proc', u'natl', u'sci', u'al', u'biochem', u'mol', u'res', u'biophys', u'commun', u'acad',
               u'chem', u'med', u'biol', u'enzymol', u'Am', u'Soc', u'pat', u'nos', u'id', u'seq',
               u'gen', u'ed', u'publ', u'cell', u'ii', u'iii', u'iv', u'viral', u'dis', u'infect',
               u'rev', u'supp', u'dev', u'pp', u'genet', u'pp', u'nucl', u'pub', u'etc', u'virol',
               u'u.s. pat', u'u.s. ser', u'u.s. patent', u'ann', u'microbiol', u'environ', u'U.S',
               u'curr', u'vol', u'enz', u'struct', u'exp', u'approx', u'int', u'oncol', u'appl',
               u'math', u'adv', u'u.s.c', u'et', u'app', u'biosci', u'molec']
extra_abbrv.extend([str(i) for i in range(1, 40)])
extra_abbrv.extend(list('abcdefghijklmnopqrstuvwxyz'))

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentence_tokenizer._params.abbrev_types.update(extra_abbrv)

abbrev_types_set = set(sentence_tokenizer._params.abbrev_types)

def stemtokenizer(text):
    """ MAIN FUNCTION to get clean stems out of a text. A list of clean stems are returned """
    tokenizer = RegexpTokenizer(r'\s+', gaps=True)
    tokens = tokenizer.tokenize(text)
    stems = []  # result
    for token in tokens:
        stem = token.lower()
        stem = stem.strip(string.punctuation)
        if stem:
            if is_number(stem):
                stem = NUMBER_INDICATOR
            elif is_currency(stem):
                stem = CURRENCY_INDICATOR
            elif is_chemical(stem):
                stem = CHEMICAL_INDICATOR
            else:
                stem = stem.strip(string.punctuation)
            if stem and len(stem) >= MIN_SIZE:
                # extract uni-grams
                stems.append(stem)
    del tokens
    return stems

punctuation_to_strip = u'"#%&\'();:*+-/<=>@[\\]^_`{|}~'
punctuation_to_strip += u'\u2018\u2019\u201c\u201d' # additional punctuation not in string.punctuation
# \u2018 => ‘ \u2019 => ’ \u201c => “ \u201d => ”
def sentence_wordtokenizer(text):
    """
    Improved function for tokenization, stripping only specific punctuation and preserving the punctuation
    . , : ; ? ! as separate tokens
    """
    tokenizer = RegexpTokenizer(r'\s+', gaps=True)
    tokens = tokenizer.tokenize(text)
    stems = []  # result
    for token in tokens:
        stem = token.lower()
        stem = stem.strip(punctuation_to_strip)
        if stem:
            if is_number(stem):
                token_stems = [NUMBER_INDICATOR]
            elif is_currency(stem):
                token_stems = [CURRENCY_INDICATOR]
            elif is_chemical_new(stem):
                token_stems = [CHEMICAL_INDICATOR]
            else:
                token_stems = list(re.findall(r"^([.,!?;:]*)(.+?)([.,!?;:]?)$", stem)[0])
                token_stems = [t.strip(punctuation_to_strip) for t in token_stems]
                if len(token_stems):
                    if token_stems[1] in abbrev_types_set:
                        token_stems[1] = token_stems[1] + token_stems[2]
                        token_stems[2] = ""

            for stem in token_stems:
                if stem and len(stem) >= MIN_SIZE:
                    # extract uni-grams
                    stems.append(stem)
    del tokens
    return stems


def is_number(str):
    """ Returns true if given string is a number (float or int)"""
    try:
        float(str.replace(",", ""))
        return True
    except ValueError:
        return False

def is_currency(str):
    return str[0] == "$"

def is_chemical(str):
    return str.count("-") > 3

def is_chemical_new(strg):
    return (strg.count("-") > 2 and len(strg) >= 25) or len(strg) >= 40

def is_table_row(sent):
    return len(sent) < MAX_NON_TABLE_ROW_LENGTH and sent.count('.') > 2

def min_sentence_length_enforcer(sentences):
    # initialize with '' so we dont have to add a condition to be evaluated in the loop
    new_sentences = ['']
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 0:
            if len(sent) < MIN_SENT_LENGTH or is_table_row(sent):
                new_sentences[-1] += ' ' + sent
            else:
                new_sentences.append(sent)

    # handling of special cases where the first sentence is either kept as '' or is something like '1. '
    if len(new_sentences[0]) == 0:
        # takes care of the case where there are no sentences other than ''=> new_sentences = [] and
        # when there are => new_sentences = new_sentences[1:]
        new_sentences = new_sentences[1:]
    elif len(new_sentences[0]) < MIN_SENT_LENGTH and len(new_sentences) > 1:
        new_sentences[1] = new_sentences[0] + ' ' + new_sentences[1]
        new_sentences = new_sentences[1:]
    return new_sentences

def get_sentences(text):
    sents = sentence_tokenizer.tokenize(text)
    sents = min_sentence_length_enforcer(sents)
    return sents
