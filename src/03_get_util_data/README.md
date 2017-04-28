### Purpose

Reads from the elasticsearch database and writes out the following information to pickle files to be use later:
 
 - Names of sections, classes and subclasses
 - list of valid classes and subclasses that meet the minimum number of documents limit
 - an index for documents to associate every document with its classifications
 - an index for classficiations, to associate every classification (whether section, class or subclass) with the documents having it
 - training, validation and test document ids
 
 This script uses spark to download data from elasticsearch, so a spark instance running locally is required