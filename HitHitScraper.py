import requests 
from bs4 import BeautifulSoup
import pickle
import time
import socket


targetFile = "HitHit.pickle"
def loadPcl(savedFile): 
        with open(savedFile, "rb") as file:
            return pickle.load(file)
        
def save(object):
    try:
        with open(targetFile,"wb") as file:
            pickle.dump(object, file, protocol= pickle.HIGHEST_PROTOCOL)        
    except Exception as ex:
        print("File was not saved: ", ex)

def getWebsiteText(adress):
    try:
        htmlText = requests.get(adress).text    #the .text ending = we accept only text, not status code
        return htmlText
    except:
        print("Chybí připojení k internetu.")
        return "Nestaženo"
    
def getWebsite(adress):
    wait_for_connection()
    htmlText = getWebsiteText(adress)

    while check_forbidden(htmlText):
        print("403 Forbidden detected. Waiting for 5 seconds...")
        time.sleep(5)
        htmlText = getWebsiteText(adress)

    
    stranka = BeautifulSoup(htmlText, "html.parser")  # "lxml" či "html.parser" značí algoritmus, kterým chceme s kódem zacházet
    return stranka 

def is_connected():
    try:# Try to connect to a well-known host (Google's DNS server)
        socket.create_connection(("8.8.8.8", 53))
        return True
    except OSError:return False


def wait_for_connection():
    print("Waiting for internet connection...")
    while not is_connected():
        time.sleep(5)   # Wait for 5 seconds before checking again


# Define the HTML content to check for
forbidden_html = """
<head><title>403 Forbidden</title></head>
<body>
<center><h1>403 Forbidden</h1></center>
</body>
</html>
"""


# Function to check for "403 Forbidden" in the string
def check_forbidden(html_content):
    return forbidden_html in html_content



pages = []
stop = 15000
start = 0
for i in range(start,stop):
    adress = f"https://www.hithit.com/cs/project/{i}"
    page = getWebsite(adress)
    print(f"Page {i} parsed.")
    #if "Promiňte, přístup byl zamítnut. (403)" in str(page):print(f"Page {i} access denied.")
    if "AJAJ, NĚKDE SE STALA CHYBA. (505)" in str(page): print("It seems we are out of range.")
    else: 
        pages.append((i, adress, page))
        print(f"Page {i} appended.")
    print(f"Progress: {round((i-start)/(stop-start)*100,2)} %")

try: 
    save(pages)
    print("Pages succesfully saved")
except: 
    print("Error when saving the files")
