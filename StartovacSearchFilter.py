
import pickle

targetFile = r"LinksStartovac.pickle"

def save(object):
    try:
        with open(targetFile,"wb") as file:
            pickle.dump(object, file, protocol= pickle.HIGHEST_PROTOCOL)        
    except Exception as ex:
        print("File was not saved: ", ex)

def loadPcl(savedFile): 
        with open(savedFile, "rb") as file:
            return pickle.load(file)

def getLinks(page):
    all_links = page.find_all('a', href=True)
    project_links = [link['href'] for link in all_links if link['href'].startswith('/projekty/')]
    return(project_links)

    #returns a list with links

# script itself: 

pages = loadPcl(r"LinksStartovacRaw.pickle")
print(type(pages))
print(type(pages[1]))
print(type(pages[1][3]))
print(pages[1][3].prettify())

print("Link extraction running")
new_links = []
links = loadPcl(targetFile)
for page in pages:
    new_links = new_links + getLinks(page[3])
    print(f"Extraction progress: page {page[0]} of {len(pages)}. {round((page[0]+1)/(len(pages))*100,2)} %")

links = links + new_links
unique = list(set(links))
print(f"The number of gained new links is {len(new_links)}. The number of unique links is {len(unique)}")
high_score = 2827               #2726 
total = 2758
if len(unique) > high_score:
    print(f"New links found! The number of missing links decreased from {total-high_score} to {total-len(unique)}.")
save(unique)
print("End of program.")