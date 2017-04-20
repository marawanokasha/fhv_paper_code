import Data_Handler as dataHandler
import Parser as parser
import os
from pprint import pprint
import pickle

path_to_data = os.path.abspath(os.path.dirname(__file__)) + "/Data/"
if not os.path.exists(path_to_data):
    os.makedirs(path_to_data)
for year in list(range(2006, 2016)):
    dataHandler.download_patent_files(year)

# expects an elasticsearch instance at localhost port 9200
# creates the index "patents"
parser.parse_and_upload_data()    
