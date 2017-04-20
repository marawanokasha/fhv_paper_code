import sys, os
from io import StringIO
import re
from bs4 import BeautifulSoup
from lxml import etree
import logging
import datetime
from multiprocessing import Pool, Manager
from elasticsearch import Elasticsearch
import io
import os.path
from zipfile import ZipFile
from pprint import pprint

logging.basicConfig(filename='Data_management_files/error-log.log', level=logging.ERROR)

tags = {4 : {
                 
                 # BASIC PATENT DATA
                 "patent-country" : "/us-patent-grant/us-bibliographic-data-grant/publication-reference/document-id/country",
                 "patent-doc-number" : "/us-patent-grant/us-bibliographic-data-grant/publication-reference/document-id/doc-number",
                 "patent-date" : "/us-patent-grant/us-bibliographic-data-grant/publication-reference/document-id/date",
                 "patent-kind" : "/us-patent-grant/us-bibliographic-data-grant/publication-reference/document-id/kind",
                 "application-country":"/us-patent-grant/us-bibliographic-data-grant/application-reference/document-id/country",
                 "application-doc-number" : "/us-patent-grant/us-bibliographic-data-grant/application-reference/document-id/doc-number",
                 "application-date" : "/us-patent-grant/us-bibliographic-data-grant/application-reference/document-id/date",
                 "inventors" : "/us-patent-grant/us-bibliographic-data-grant/parties/applicants//applicant/addressbook//*[self::last-name | self::first-name |self::city |self::country] |"  
                                + "/us-patent-grant/us-bibliographic-data-grant/us-parties/us-applicants//us-applicant/addressbook//*[self::last-name | self::first-name |self::city |self::country] ",
                
                # TITLE, ABSTRACT, DESCRIPTIONS AND CLAIMS
                 "invention-title": "/us-patent-grant/us-bibliographic-data-grant/invention-title",
                "abstract" : "/us-patent-grant/abstract",
                "description" : "/us-patent-grant/description",
                "claims" : "/us-patent-grant/claims",
                
                # CLASSIFICATION DATA
                "classification-national-main": "/us-patent-grant/us-bibliographic-data-grant/classification-national/main-classification",
                "classification-national-further-classification": "/us-patent-grant/us-bibliographic-data-grant/classification-national//further-classification",
                "classification-ipc":"/us-patent-grant/us-bibliographic-data-grant/classification-ipc/*[self::main-classification | self::further-classification] |" + 
                                        "/us-patent-grant/us-bibliographic-data-grant/classifications-ipcr//classification-ipcr/*[self::section | self::class| self::subclass | self::main-group | self::subgroup]",
                "us-field-of-classification-search" : "/us-patent-grant/us-bibliographic-data-grant//classification-national/main-classification",
                 
                "references-cited": "/us-patent-grant/us-bibliographic-data-grant/references-cited//citation/patcit/document-id/doc-number" + 
                                        "|/us-patent-grant/us-bibliographic-data-grant/us-references-cited//us-citation/patcit/document-id/doc-number",
                 },
2 : {
                 "patent-country":"/PATDOC/SDOBI/B100/B190/PDAT",
                 "patent-doc-number": "/PATDOC/SDOBI/B100/B110/DNUM/PDAT",
                 "patent-date": "/PATDOC/SDOBI/B100/B140/DATE/PDAT",
                 "patent-kind": "/PATDOC/SDOBI/B100/B130/PDAT",
                 "application-country":"/PATDOC/SDOBI/B100/B190/PDAT",
                 "application-doc-number": "/PATDOC/SDOBI/B100/B110/DNUM/PDAT",
                 "application-date": "/PATDOC/SDOBI/B100/B140/DATE/PDAT",
                 "inventors":"/PATDOC/SDOBI/B700/B720//B721//*",
                 
                 "invention-title": "/PATDOC/SDOAB/BTEXT/PARA/PTEXT/PDAT",
                 "abstract": "/PATDOC/SDOAB",
                 "description": "/PATDOC/SDODE",
                 "claims": "/PATDOC/SDOCL",
                
                 "classification-national-main": "/PATDOC/SDOBI/B500/B520/B521/PDAT",
                 "classification-national-further-classification": "/PATDOC/SDOBI/B500/B520//B522/PDAT",
                 "classification-ipc":"/PATDOC/SDOBI/B500/B510/B511/PDAT | " + 
                                        "/PATDOC/SDOBI/B500/B510//B512/PDAT",
                 "us-field-of-classification-search" : "/PATDOC/SDOBI/B500/B580/B582//*",
                                        
                 "references-cited":"/PATDOC/SDOBI/B500/B560//B561/PCIT/DOC/DNUM/PDAT"                      
}}


####__________PROCESS_PATENT__________####
def process_patent(patent, dtd_version):
    # remove all useless values which might interfere with parsing the xml file
    patent = re.sub("(<\?(?!xml).+?\?>|<b>|<\/b>|<i>|<\/i>|<figref.*?>|<\/figref>|"
        + "<sequence-cwu.*?>|<\/sequence-cwu>|<claim-ref.*?>|<\/claim-ref>)", "", patent)
    patent_data = {}
    try:
        soup = BeautifulSoup(patent, "xml")
        f = StringIO(str(soup.contents[len(soup.contents) - 1]))
        tree = etree.parse(f)
        
        # go through all the tag names
        for tag_name in tags[dtd_version]:
            
            # try to get the values for each tag name by using the XPath command specific 
            # to the dtd-version of the patent 
            values = tree.xpath(tags[dtd_version][tag_name])
            # if all the values for the tag name were found, save their text representation as a list 
            if not (values == []):
                if tag_name in ('abstract', 'description', 'claims'):
                    patent_data[tag_name] = [str(etree.tostring(values[0], encoding='utf8', method='xml'), 'utf8')]
                else:
                    patent_data[tag_name] = [value.text for value in values]
    
        # organize the processed patent data
        patent_data = organize_processed_patent(patent_data, dtd_version)
        if(patent_data is not None):
            upload_data(patent_data)
            return patent_classifications(patent_data)

    except Exception as e:
        now = datetime.datetime.now()
        logging.debug("The Following parsing error occurred while parsing patent: \n" + str(e) + "(" + now.strftime("%Y-%m-%d %H:%M:%S") 
                        + ")\n_______________________________________________________________________\n")
        print("new parsing error occurred")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return None
    
def patent_classifications(patent):
    classifications = []
    for classification in patent["classification-ipc"]:
        classifications.append(classification["section"] + " " + patent["patent-doc-number"] + "\n" + 
                            classification["class"] + " " + patent["patent-doc-number"] + "\n" + 
                            classification["subclass"] + " " + patent["patent-doc-number"] + "\n")
    return classifications
        

####__________ORGANIZE_PROCESSED_PATENT__________####
def organize_processed_patent(patent, dtd_version):
    
    new_patent = {}
    
    # if the patent does not have an ipc-classification it cannot be used for
    # classification and is therefore removed and no longer processed
    if("classification-ipc" not in patent.keys() or 
       "inventors" not in patent.keys() or 
       "abstract" not in patent.keys() or 
       "claims" not in patent.keys() or
       "description" not in patent.keys()):
        return None
    try:
        
        # go through all the values for each tag name of the patent 
        for tag_name, values in patent.items():
            new_patent[tag_name] = []
            proccesed_values = []
            for val in values:
                # remove newline, empty and None entries
                if (type(val) != str or not re.match("(^\\n)", val)) and val is not None:
                    if(re.match("^classification", tag_name) or tag_name == "references-cited" or tag_name == "us-field-of-classification-search"):
                        val = re.sub("\s+?", "", val)
                    proccesed_values.append(val)
                    new_patent[tag_name].append(val)
            
            # save each ipc-classification of the patent as a list of dictionaries. each dictionary containing 
            # it's secition, class and subclass value
            if(tag_name == "classification-ipc"):
                if(dtd_version == 2):
                    for value in proccesed_values:
                        if not re.match("^[A-Z].*", value):
                            return None
                values_text = "".join("".join(new_patent[tag_name]).split())
                new_patent[tag_name] = [{"section": x[0], "class": x[1:3], "subclass": x[3]} for x in re.findall("([A-H][0-9]{2}[A-Z][0-9]{2,4})", values_text)]
            
            # save each inventors of the patent as a dictionary containing: firstname,lastname,city,country  
            if(tag_name == "inventors"):
                inventor_tag_values = ["first-name", "last-name", "city", "country"] if dtd_version == 2 else ["last-name", "first-name", "city", "country"] 
                new_patent[tag_name] = [dict([(inventor_tag_values[x % 4], new_patent[tag_name][x]) for x in range(0, len(new_patent[tag_name]))][y * 4:(y + 1) * 4]) for y in range(0, int(len(new_patent[tag_name]) / 4))]
            
            # tag names that don't have more than one value are changed from a list to a single value   
            if(tag_name in ["invention-title", "classification-national-main",
                            "application-date", "application-doc-number", "application-country", "patent-country", "patent-date", "patent-kind",
                            "patent-doc-number"]):
                try:
                    new_patent[tag_name] = new_patent[tag_name][0] 
                except:
                    new_patent[tag_name] = ''
        
        # add ipc_classification data of the patent to root
        return new_patent
      
    except Exception as e:
        now = datetime.datetime.now()
        logging.debug("The Following error occurred while parsing patent: \n" + str(e) + "(" + now.strftime("%Y-%m-%d %H:%M:%S") 
                            + ")\n_______________________________________________________________________\n")
        print("new error occurred")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return None
    
####__________UPLOAD_DATA__________####
def upload_data(processed_data):
    
    # upload the processed patent using elastic search
    Elasticsearch([{"host" : 'localhost', "port" : "9200"}]).index(index='patents', doc_type='patent', id=processed_data["patent-doc-number"], body=processed_data)

####__________GET_PATENTS__________####
def get_patents(content, year):
    start_tag = "<?xml" 
    start_index = content.find(start_tag, 0)
    patents = []
    
    # while all the patents haven't been found (indicated by start value)
    while True:
        
        # find the next starting tag ("<?xml")
        start = content.find(start_tag, start_index)
        
        # if the next starting tag cannot be found all patents in the xml file have been found.
        # return the list of all the patents contained in the xml file
        if start == -1:
            return [(patent, 4 if year > 2004 else 2) for patent in patents]
        
        # if the next start tag was found, find the next start tag and save the value to end.
        #  append the text from start to end. then set the start to end. 
        else:
            end = content.find(start_tag, start_index + len(start_tag))
            if end == -1:
                end = len(content)
            patents.append(content[start : end])
            start_index = end

####__________EXTRACT_PATENT_ZIP_FILE__________####
def extract_patent_file(zip_file_path):
    
    zip_file = ZipFile(zip_file_path)
    xml_index = -1
    
    # Go through all the files within a .zip file.     
    for x in range(0, len(zip_file.infolist())):
        
        # if the file is an xml file save the index of that file
        if re.search("XML|xml", zip_file.infolist()[x].filename):
            xml_index = x
            break
    if xml_index == -1:
        return None
    
    # access the xml file by using it's index in the zip file and return a TextIOWrapper
    # representation of it
    items_file = zip_file.open(zip_file.infolist()[xml_index].filename)
    return io.TextIOWrapper(io.BytesIO(items_file.read()))

####__________SAVE_CLASSIFICATION_DICTIONARY__________####
def save_classification_dictionary_to_file(xml_root):
    
    # save xml file from to file 'classification_dictionary.xml'
    tree = etree.ElementTree(xml_root)
    tree.write("Data_management_files/classification_dictionary.xml")
        
        
def listener(q):

    f = open("Data_management_files/classification_dictionary.txt", 'a') 
    while 1:
        m = q.get()
        if m == 'kill':
            break
        f.write(str(m))
        f.flush()
    f.close()
    
####__________PARSE_AND_UPLOAD_DATA__________####
def parse_and_upload_data():
    path = os.path.dirname(__file__) + "/Data/"
    pool = Pool()
    m = Manager()
    q = m.Queue()
    pool.apply_async(listener, (q,))
    # Go through all the files in Data directory which contains all the patent files 
    for (path, dirs, file_names) in os.walk(path):
        if re.search("([0-9]{4})", path):
            for patent_file_name in file_names:
                    print(patent_file_name)
                    #if the file is a .zip file then open it.
                    if re.search("(zip)", patent_file_name):
                        patent_file = extract_patent_file(os.path.join(path, patent_file_name))
                        pprint(patent_file_name)
                        if patent_file is not None:
                             
                            # retrieve all the patents saved within the file
                            patents = get_patents(patent_file.read(), int(re.search("([0-9]{4})", path).group()))
                            
                            results = [result.get() for result in [pool.apply_async(process_patent, (patent, dtd_version,)) for patent, dtd_version in patents] if result.get() is not None]
                            for result in results:
                                for value in result:
                                    q.put(value)
                    
    q.put("kill")
