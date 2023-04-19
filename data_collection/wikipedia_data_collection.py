import requests
import os
import time
from bs4 import BeautifulSoup

# Set up parameters for API query
params = {
    "action": "query",
    "list": "categorymembers",
    "cmtitle": "Category:WikiProject_Food_and_drink_articles",
    "cmlimit": "500",
    "format": "json"
}

# Initialize list to hold page titles
page_titles = []

# Loop through query with continuation until all pages are fetched
while True:
    # Fetch pages with current parameters
    response = requests.get("https://en.wikipedia.org/w/api.php", params=params)
    data = response.json()
    
    # Extract page titles and add them to the list
    for page in data["query"]["categorymembers"]:
        page_titles.append(page["title"])
    
    # Check if there are more pages to fetch
    if "continue" in data:
        params["cmcontinue"] = data["continue"]["cmcontinue"]
    else:
        break

# Create directory to store articles
if not os.path.exists("articles"):
    os.makedirs("articles")

articles_so_far = 0
# Loop through pages and fetch article text
for title in page_titles:
    # Prints number of articles fetched so far so we know it's not just hanging.
    articles_so_far += 1
    if articles_so_far % 50 == 0:
        print(articles_so_far)
    # Set up parameters for API query
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "format": "json",
        "explaintext": True
    }
    
    # Fetch article text
    response = requests.get("https://en.wikipedia.org/w/api.php", params=params)
    data = response.json()
    
    # Get article text and clean using BeautifulSoup
    article_text = list(data["query"]["pages"].values())[0]["extract"]
    soup = BeautifulSoup(article_text, "html.parser")
    cleaned_text = soup.get_text()
    
    # Save cleaned article text to file
    with open("articles/" + title.replace("/", "") + ".txt", "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    time.sleep(0.1)