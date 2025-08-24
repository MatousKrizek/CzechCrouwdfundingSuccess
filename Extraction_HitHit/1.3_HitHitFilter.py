from bs4 import BeautifulSoup
import pickle
import re
import pandas as pd
import os 

def loadPcl(savedFile): 
        with open(savedFile, "rb") as file:
            return pickle.load(file)
targetFile = r"HitHitDF.pickle"        
def save(object):
    try:
        with open(targetFile,"wb") as file:
            pickle.dump(object, file, protocol= pickle.HIGHEST_PROTOCOL)        
    except Exception as ex:
        print("File was not saved: ", ex)
        
def f1(string):
    if ">" in string:  
        for i in range(0,len(string)):
            if string[i] == ">":
                    leave=""
                    for j in range(i+1, len(string)):
                        if string[j] == "<": 
                            return leave
                            break
                        leave = leave+string[j]
        

    else: return string                          
                     
def fDtN(string):
    """This function filters out everything except for numerals out of the text string.""" 
    result = ""
    string=str(string)
    dirty = string.replace("tis.", "000")
    dirty = dirty.replace("tis","000")
    dirty = dirty.replace("mil.","000000")
    dirty = dirty.replace("mil","000000")
    for letter in dirty:
        if letter in ["1","2","3","4","5","6","7","8","9","0","-"]:
            result= result + letter
        elif letter in ["."]:       # integrated translate to decimal commas
            result= result + ","
    return result

def purge_code_remains(text):
    text = text.replace('\xa0', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')  
    return(text)

source_file = r"HitHit.pickle"
print(os.path.abspath(source_file))
      
raw_data = loadPcl(source_file)
del(raw_data[0])        # there is an error page on position 0 hiding


def mine(h):
    """
    Mines infromation from a scriped HitHit project.
    Input: h = touple(project number, project page link, project page as bs4 data type)
    
    """
     
    print(h)
    link = h[1]
    print(link)
    s = h[2]                        #selects the bs4 file containing the web page
    title =s.h1.text.strip()
    
    authorDirt = s.find("div", {"class": "projectAuthor"}).text     
    author=authorDirt.replace("Autor:","").strip()
    print("author reached")

    description = s.find("div", {"class": "pg-projectcopy-main"}).text  # we get complete text inc. "Více..."
    description = purge_code_remains(description)
    print("Description reached")
    mode = s.find_all("a", {"href": "https://www.hithit.com/cs/article/whatIsHithit"})[-2].text
    end_date = extractHHDate(s)
    print("End date reached")
    money_main = s.find("div",{"class": "pg-project-main-stats-in"}).find_all("span", {"class":"currency"})                 
    # money_main obsahuje peněžní částky z hlavních informací = získanou [0] a cílovou [1] částku
    collected = float(fDtN(money_main[0].text))
    goal = float(fDtN(money_main[1].text))
    #s_rate = float(collected)/float(goal)
    rewards = extractHHRewards(s)
    reward_titles, reward_descr, reward_prices = zip(*rewards)

    #success = s.find("span", {"class": "label"}).text
    s_rate2 = s.find("div", {"class": "progressPercentage"}).text      # převezme success rate přímo z HTML
    n_contributed = int(s.find_all("strong")[1].text)
    categoriesDirty = s.find("div",{"class": "pg-project-main-stats-in"}).find_all("a", {"class":""})
    categories = [None for i in range(len(categoriesDirty)-1)]
    for i in range(0,len(categoriesDirty)-1):                    # chceme se vyhnout zapojení "All or nothing"
        categories[i] = categoriesDirty[i].text
    try: location = s.find("div",{"class": "pg-project-main-stats-in"}).find("p",{"class":"locationBox"}).text.strip()   
    except: location = "NA" 
    videos = count_videos(s)
    return({
        "Title": str(title), 
        "Author":str(author), 
        "Link": str(link), 
        "Description":str(description), 
        "Mode":str(mode), 
        "End_date":end_date,                    # to be converted
        "Amount collected":collected, 
        "Goal":goal, 
        "Success rate":str(s_rate2), 
        "N_contributors":int(n_contributed), 
        "Location":str(location), 
        "Categories":list(categories), 
        "Reward_titles":list(reward_titles), 
        "Reward_descrs":list(reward_descr), 
        "Reward_prices":list(reward_prices),
        "N_videos": int(videos),
        "Platform": "Hithit"})
    """
    rewards =
    projecLinks = []

    #per reward: 
    name =
    price =
    capacity =
    orders =
    drawed = # %
    desc = 
    delivery =
    """

#for h in raw_data:
    print(f"The type of each data item is f{type(h)}")
    for i in h: 
        print(f"The type of {h.index(i)}-th element is {type(i)}")
        if len(str(i)) < 100: 
            print(f"and the value is {i}")
    # We identified, that the raw_data is a list of touples of three elements - the ordinal number (0-98)
    # of the project, the link to the project page, and the bs4 file with the project page
    # The sample list containt 47 projects

# New (chat gpt) version
def extractHHDate(HH_bs4):
    pattern = r'Projekt skončil (\d{1,2}\.\d{1,2}\.\d{4} v \d{2}:\d{2})'

    for p in HH_bs4.find_all('p'):
        match = re.search(pattern, p.text, re.IGNORECASE)
        if match:
            date_text = match.group(1).strip()
            numeric_date = re.sub(r'[^\d\.\s:]', '', date_text)
            return numeric_date
    return "No date found"

#♥def extractHHDate(HH_bs4):
 # Define the combined regex pattern
    pattern = (
        r'Projekt skončil  (\d{2}\.\d{2}\.\d{4} v \d{2}:\d{2})|'
        r'Projekt skončil (\d{2}\.\d{1}\.\d{4} v \d{2}:\d{2})|'
        r'Projekt skončil (\d{1}\.\d{2}\.\d{4} v \d{2}:\d{2})|'
        r'Projekt skončil (\d{1}\.\d{1}\.\d{4} v \d{2}:\d{2})'
    )
    
    # Find all p tags and search for the specific text pattern
    for p in HH_bs4.find_all('p'):
        match = re.search(pattern, p.text)
        if match:
            date_text = match.group().strip()
                    # Filter out all letters and keep only the numeric part, spaces, dots and double dots
            numeric_date = re.sub(r'[^\d\.\s:]', '', date_text)
            return numeric_date
    return "No date found"

def extractHHRewards(HH_bs4):
    """
    Finds all rewards in a HitHit project page and returns their names and prices.
    """
    rewards = HH_bs4.find_all("div", class_="pg-rewards-i")
    reward_details = []
    for reward in rewards:
        # Extract the reward name
        name_div = reward.find("h3", class_="rewardTitle")
        if name_div:
            # Get all non-empty text elements within the h3 tag
            name_parts = [text for text in name_div.find_all(text=True, recursive=False) if text.strip()]
            # Select the last non-empty text element as the reward name
            name = name_parts[-1].strip() if name_parts else None
            #name = purge_code_remains(name)
        else:
            name = None
        
        # Extract the reward description
        description_div = reward.find("div", class_="pg-rewards-i-content")
        description = description_div.get_text(strip=True) if description_div else None
        description = purge_code_remains(description)
        
        # Extract the reward price
        price_span = reward.find("span", class_="currency")
        if price_span:
            price_text = price_span.string.strip()
            # Remove any non-numeric characters (e.g., currency symbols)
            numeric_text = re.sub(r'[^\d.,]', '', price_text)
            # Replace commas with dots if necessary (for decimal points)
            numeric_text = numeric_text.replace(',', '.')
            # Convert to float
            price = float(numeric_text)
        
        # Append the reward name and price to the list for each unique description
        reward_details.append((name,description,price))
        
    #purge duplicates by unique description
    
    return sorted(list(set(reward_details)), key = lambda x: x[2])   #after removing the duplicates by set(), we re-sort based on 3rd element

def detect_video(bs4):
    """This function check for the presence of a YouTube or Vimeo video on the website"""
# Check for common video tags
    video_tags = bs4.find_all('video')
    iframe_tags = bs4.find_all('iframe')
    embed_tags = bs4.find_all('embed')
    object_tags = bs4.find_all('object')

    # Check if any of the tags contain video content
    if video_tags:
        print("Video tag found.")
        return True
    for iframe in iframe_tags:
        if 'youtube' in iframe.get('src', '') or 'vimeo' in iframe.get('src', ''):
            print("YouTube or Vimeo iframe found.")
            return True
    for embed in embed_tags:
        if 'youtube' in embed.get('src', '') or 'vimeo' in embed.get('src', ''):
            print("YouTube or Vimeo embed found.")
            return True
    for obj in object_tags:
        if 'youtube' in obj.get('data', '') or 'vimeo' in obj.get('data', ''):
            print("YouTube or Vimeo object found.")
            return True

    print("No video found.")
    return False

def count_videos(bs4):
    soup = bs4
    video_count = 0
    # Count <video> tags
    video_tags = soup.find_all('video')
    video_count += len(video_tags)

    # Count <iframe> tags with video URLs
    iframe_tags = soup.find_all('iframe')
    for iframe in iframe_tags:
        if 'youtube' in iframe.get('src', '') or 'vimeo' in iframe.get('src', ''):
            video_count += 1

    # Count <embed> tags with video URLs
    embed_tags = soup.find_all('embed')
    for embed in embed_tags:
        if 'youtube' in embed.get('src', '') or 'vimeo' in embed.get('src', ''):
            video_count += 1

    # Count <object> tags with video URLs
    object_tags = soup.find_all('object')
    for obj in object_tags:
        if 'youtube' in obj.get('data', '') or 'vimeo' in obj.get('data', ''):
            video_count += 1

     # Find all <a> tags with class "playButton" and data-video-url attribute
    video_links = soup.find_all('a', class_='playButton', attrs={'data-video-url': True})

    # Count the number of video links found
    video_count = video_count + len(video_links)
    print(f"The number of videos in file: {video_count}")
    return video_count


# The main script: 
mined_data = []
failed_minings = []
n = len(raw_data)
for i in range(n):
    print(f"File {i}/{n} opened")
    try: 
        mined_data.append(mine(raw_data[i]))                    # I suspect HitHit of removing some campaign from being publically visible
        print(f"File {i}/{n} mined")
    except: 
        print(f"Unable to mine the link {raw_data[i][1]}")
        failed_minings.append(raw_data[i][:1])

print(f"{len(mined_data)} out of {n} files succesfully mined ({round(len(mined_data)/n*100,2)} %)")

# Final dataframe
df = pd.DataFrame(mined_data)
save(df)

print("First few rows of the DataFrame:")
print(df.head())

# Display summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Display information about the DataFrame (including non-null counts and data types)
print("\nInformation about the DataFrame:")
print(df.info())

# Check for missing values in the DataFrame
print("\nMissing values in the DataFrame:")
print(df.isnull().sum())

# Display unique values in each column
print("\nNumber of unique values in each column:")
for column in df.columns:
    try: print(f"{column}: {len(df[column].unique())}")
    except: print(f"Type of column {column} does not allow to calculate unique values")

print("List of failed minings follows:")
print(failed_minings)
