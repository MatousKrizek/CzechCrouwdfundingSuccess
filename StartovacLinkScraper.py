import pickle
targetFile = r"LinksStartovacRaw.pickle"
from bs4 import BeautifulSoup 
def save(object):
    try:
        with open(targetFile,"wb") as file:
            pickle.dump(object, file, protocol= pickle.HIGHEST_PROTOCOL)        
    except Exception as ex:
        print("File was not saved: ", ex)


from selenium import webdriver                                             
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By 
import time

options = Options()
options.add_argument("--disable-search-engine-choice-screen")

#skrytí automatizace před spuštěním
options.add_argument("--start-maximized")
options.add_argument('disable-infobars')
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.page_load_strategy = 'eager'

# options.headless = True
b = webdriver.Chrome(options=options)

# skrytí automatizace po spuštění
b.delete_all_cookies()
b.execute_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined })")
b.get("httpes://www.google.com")

url = "https://www.startovac.cz/hledat?q=a"
b.get(url)

all = False
nClick=0
while not all:                    
    try: 
        time.sleep(0)
        #print("Jdeme do problému")
        b.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        btn = b.find_element(By.XPATH,"//button[contains(text(),'Zobrazit další')]")
        #print("Jdeme z problému")
        print(type(btn))
        btn.click()
        nClick +=1
        print(f"Button clicked {nClick} times.")
        time.sleep(3)
        
    except NoSuchElementException: 
        print("Clicking done")
        all = True

page = b.page_source
#print(page)
save(BeautifulSoup(page, "html.parser"))
#b.quit()
print("Script finished")
