### Purpose

Reads from the file containing the entire documents as created by the previous step and then creates the BOW matrices used by classifiers

This script uses spark to parallelize the reading from the file, so a spark instance running locally is required