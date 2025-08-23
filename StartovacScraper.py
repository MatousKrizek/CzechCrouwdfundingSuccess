# macro parameters
folder_of_saving = r"SavingFolder"
linksFile = r"LinksStartovac.pickle"
chrome_profile_path = r"C:\Users\User\AppData\Local\Google\Chrome\User Data"

start_session = 17           # 0-based number of the first session to run # start with 10 
n_sessions_to_run = 3
pages_per_session = 100





import requests 
from bs4 import BeautifulSoup
import pickle
import math
import sys

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time

def loadPcl(savedFile): 
        with open(savedFile, "rb") as file:
            return pickle.load(file)
        
def save(object):
    try:
        with open(targetFile,"wb") as file:
            pickle.dump(object, file, protocol= pickle.HIGHEST_PROTOCOL)        
    except Exception as ex:
        print("File was not saved: ", ex)

def save2(object, target_file):
    try:
        with open(target_file,"wb") as file:
            pickle.dump(object, file, protocol= pickle.HIGHEST_PROTOCOL)        
    except Exception as ex:
        print("File was not saved: ", ex)

def getWebsite(adress):
    driver = webdriver.Chrome()
    
    try:
        # Načtení stránky
        driver.get(adress)

        # Čekání na načtení potřebného obsahu (např. tag <abbr> s třídou "UFISutroCommentTimestamp livetimestamp")
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "UFISutroCommentTimestamp"))
            )
            # Přidání dalšího čekání pro jistotu
            time.sleep(2)  # čekání 2 sekundy
        except:
            print("Tag not found, but proceeding with page download.")
        finally:
            # Získání obsahu stránky po načtení
            htmlText = driver.page_source
            driver.quit()
                #htmlText = requests.get(adress).text    #the .text ending = we accept only text, not status code
                #print(htmlText)  # teď je webovka zkopírována v html. Je to hnusná hromada bordelu a hledaných dat
    except:
        print("Chybí připojení k internetu.")
    stranka = BeautifulSoup(htmlText, "html.parser")  # "lxml" či "html.parser" značí algoritmus, kterým chceme s kódem zacházet
    return stranka 

def getSWebsite(adress):
    # příprava prohlížeče
    options = Options()
    options.add_argument("--disable-search-engine-choice-screen")
    options.add_argument(f"user-data-dir={chrome_profile_path}")

    #skrytí automatizace před spuštěním
    options.add_argument("--start-maximized")
    options.add_argument('disable-infobars')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.page_load_strategy = 'eager'

    cookies = [
    {'name': 'cookie_consent', 'value': 'true', 'domain': '.startovac.cz', 'path': '/'},
    {'name': 'cookie_analytics', 'value': 'true', 'domain': '.startovac.cz', 'path': '/'},
    {'name': 'cookie_marketing', 'value': 'true', 'domain': '.startovac.cz', 'path': '/'},
    #{'name': 'third_party', 'value': 'true', 'domain': 'startovac.cz', 'path': '/'}
]
    # podařilo se?
    failed = False
    # options.headless = True
    b = webdriver.Chrome(options=options)

    # skrytí automatizace po spuštění
    b.delete_all_cookies()

    # add new cookies 
    #for cookie in cookies:
    #    b.add_cookie(cookie)
    b.execute_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined })")
    try:
        b.get("httpes://www.google.com")
    except:
        print("Chybí připojení k internetu uvnitř stahovací funkce.")
    try: 
        b.get(adress)
        print("Adress arrived")
        if EC.element_to_be_clickable((By.XPATH, "//span[normalize-space()='Souhlasím']")):
            print("element clickable")
            time.sleep(1)
            try: 
                cookieAgree = b.find_element(By.XPATH, "//span[normalize-space()='Souhlasím']")
                cookieAgree.click()
                print("Cookies agreed.")
            except Exception as exp: 
                print(f"Cookies not allowed ")#due to: {exp}")due to: {exp}")

        if EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div[1]/div[1]/div/main/div/div[1]/div[4]/div[1]/div/button")): #/html/body/div[1]/div[1]/div[1]/div/main/div/div[1]/div[4]/div[1]/div/button
            print("Secondary cookies clickable")
            time.sleep(2)
            try: 
                cookieOpen = b.find_element(By.XPATH, "/html/body/div[1]/div[1]/div[1]/div/main/div/div[1]/div[4]/div[1]/div/button")
                time.sleep(1)
                cookieOpen.click()                      
                print("Secondary cookies opened.")
                time.sleep(1)
                try: 
                    cookie2Agree = b.find_element(By.XPATH, '//*[@id="s-all-bn"]') #//*[@id="s-all-bn"]
                    cookie2Agree.click()
                except Exception as exp:
                    print(f"Secondary cookies not agreed ")#due to: {exp}")
            except Exception as exp: 
                print(f"Secondary cookies not opened ")#due to: {exp}") 

        
        try: 
            b.execute_script("window.scrollTo(0, document.body.scrollHeight);")

             #Čekání na načtení potřebného obsahu (např. tag <abbr> s třídou "UFISutroCommentTimestamp livetimestamp")
        except: None

        # Get the main page source
        main_page_source = b.page_source

        # Parse the main page source with BeautifulSoup
        #soup = BeautifulSoup(main_page_source, "html.parser")
        
        # Find all iframes
        iframes = b.find_elements(By.TAG_NAME, "iframe")
        combined_html = main_page_source
        
        for iframe in iframes:                      # mrcha byla schovaná celou dobu v iframe
            try:    
                b.switch_to.frame(iframe)

        # Wait for a specific element within the iframe to be present
                sentence = "Facebook plugin pro komentáře"
                try:    
                    WebDriverWait(b, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'body'))
                    )
                    # Now check for the text
                    WebDriverWait(b, 10).until(
                        EC.text_to_be_present_in_element((By.TAG_NAME, 'body'), sentence)
                    )
                    print("The sentence has loaded in one of the iframes!")
                except Exception as ex:
                    print(f"Facebook plugin not located due to {ex}")
                    failed = True


                iframe_source = b.page_source
                combined_html += iframe_source
                b.switch_to.default_content()
            except Exception as ex: 
                print(f"Unable to open an iframe due to exception {ex}")
                framebreak.append(ex)

        
        # Parse the combined HTML with BeautifulSoup
        combined_soup = BeautifulSoup(combined_html, "html.parser")
        
        # Check if the element is present in the combined HTML
        abbr_tag = combined_soup.find('abbr', class_='UFISutroCommentTimestamp livetimestamp')
        if abbr_tag:
            print("Element found:", abbr_tag)
            data_utime = abbr_tag['data-utime']
            print(f"The extracted data-utime attribute is: {data_utime}")
        else:
            print("The <abbr> tag with the specified class was not found.")
       

    except:
        print("Stahovací funkce nebyla schopna získat stránku")
    b.quit()
    return combined_soup, failed 

def check_internet_connection():
    try:
        # Try to make a request to a reliable website
        requests.get('https://www.google.com', timeout=5)
        return True
    except requests.ConnectionError:
        return False

# script itself:

pages_per_session = pages_per_session
n_of_sessions_to_run = n_sessions_to_run       # number of sessions that will be run before waiting for input
increment_sessions_by = 1
failed_links = []

links = loadPcl(linksFile)[:]
already_done = start_session                # number of files of the same per session length already saved
n = math.ceil(len(links)/pages_per_session)
wait = False                    # wait for input
failed_links_file = fr"{folder_of_saving}\ProjectsStartovacLFailed.pickle"

for i in range(already_done,n):
    lower_limit = i*pages_per_session
    upper_limit = min(len(links), (i+1)*pages_per_session)
    framebreak = []
    
    session_links = links[lower_limit : upper_limit]
    session_pages = []  #list čísel linků, linků a k nim příslušných projektových stránek 
    session_file = fr"{folder_of_saving}\ProjectsStartovacSp{pages_per_session}_{i}.pickle"


    
    for link in enumerate(session_links): 
        real_num = link[0]+i*pages_per_session
        link = tuple((real_num, link[1], link[0]))

        # Check internet connection before starting the script
        while not check_internet_connection():
            print("No internet connection. Pausing...")
            time.sleep(5)  # Wait for 5 seconds before checking again
        
        
        
        print(link)             # Průšvih: dostanem se do situace, kdy Chrome přestane fungovat, třeba manuální reboot - část kampaní vynechána
        try:                    # neznáme množství nepovedených stažení asi deset linků před https://www.startovac.cz/projekty/karel-schinzel-clovek-ktery-obarvil-svet, okolo 1200 
            page, failed = getSWebsite(f"https://www.startovac.cz{link[1]}")
            print(f"https://www.startovac.cz/{link[1]}")
            #print(page)
            if failed == False: 
                number = len(session_pages) + i * pages_per_session
                session_pages.append([number, f"https://www.startovac.cz{link[1]}", page])
                print(f"Page {link[2]+1} of {str(len(session_links))} session links parsed. {round((link[2]+1)/len(session_links)*100,2)} %.\n That is {round((link[0]+1+i*pages_per_session)/(len(links)+i*pages_per_session)*100,2)} % of total.")
            else: failed_links.append(f"https://www.startovac.cz{link[1]}")

        except Exception as ex:
            print(f"Page {link[0]+1} not parsed. Exception: {ex}")
            failed_links.append([real_num, f"https://www.startovac.cz{link[1]}", None])


    print(f"Parsing of session {i} finished. Success rate: {round(len(session_pages)/len(session_links)*100, 2)} %.")
    try: 
        save2(session_pages, session_file)
        print("Session pages succesfully saved")
    except: 
        print(f"Error when saving session pages. Session number: {i}")

    try: 
        save2(failed_links, failed_links_file)
        print("Session pages succesfully saved")
    except: 
        print(f"Error when saving session pages. Session number: {i}")

    
    if i >= (n_of_sessions_to_run-1): wait = True
    
    while wait:
        if len(framebreak) > 0:
            print("Some iFrames were not accessed due to:")
            print(framebreak)
        
        key = input("Press ´x´ to start another session. Press ´f´ to quit.")
        if key == "x" or "X":
            try: 
                increment_sessions_by = int(input("How many extra sessions do you wish to make?"))
                n_of_sessions_to_run = n_of_sessions_to_run+increment_sessions_by
            except ValueError: print("Please insert an integer.")
            
            print("New session is being started.")
            break

        elif key == "f" or "F":
            key2 = input("Are you certain?")
            if key2 == "yes" or "Yes":
                print("Script terminated.")
                sys.exit()
        else:
            print("Waiting on your input.")

