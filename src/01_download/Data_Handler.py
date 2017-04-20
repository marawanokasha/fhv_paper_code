import requests, re
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os.path


####__________DOWNLOAD_PATENT_FILES__________####
def download_patent_files(year):
    
    #collect html representation of the page containing all the wanted patents
    raw_data_page = requests.get("https://www.google.com/googlebooks/uspto-patents-grants-text.html")
    soup = BeautifulSoup(raw_data_page.content,"lxml")
    
    #go through each link ("<a...>" tags) 
    for link in soup.find_all("a"):
        link_href = link.get("href")
        
	#check if redirect location("href" attribute of "<a...>" tag) 
        # points to "http://storage.googleapis.com/patents/grant_full_text/" and the year of the patents
        # is younger that the value of year
        if "http://storage.googleapis.com/patents/grant_full_text/" in link_href and int(re.search("([0-9]{4})", link_href).group()) >= year:
            
            #save the file to /Data/x where x is the year of publication of the patents
            f = urlopen(link_href)
            path_to_file = os.path.dirname(__file__) + "/Data/" + re.search("([0-9]{4})", link_href).group()
            if not os.path.isdir(path_to_file):
                os.makedirs(path_to_file)
            if not os.path.isfile(os.path.join(path_to_file, os.path.basename(link_href))):
                print('Downloading ' + str(os.path.basename(link_href)))
                open(os.path.join(path_to_file, os.path.basename(link_href)), "wb").write(f.read())       
            else:
                print('Skipping ' + str(os.path.basename(link_href)))
                



