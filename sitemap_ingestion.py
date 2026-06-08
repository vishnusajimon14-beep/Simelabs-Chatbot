import requests
from bs4 import BeautifulSoup
import pandas as pd
site_index="https://simelabs.com/sitemap.xml"
response = requests.get(site_index)
response.raise_for_status()

soup = BeautifulSoup(response.text, "xml")

sitemap_urls = [loc.text for loc in soup.find_all("loc")]

# print("Found Sitemaps:\n")

# for sitemap in sitemap_urls:
#     print(sitemap)

useful_sitemap=[]
for sitemap in sitemap_urls:
    if any(x in sitemap for x in["category-sitemap","post_tag-sitemap","author-sitemap"]):
        continue
    useful_sitemap.append(sitemap)
print("useful sitemaps")
all_url=[]
for sitemap in useful_sitemap:
    response=requests.get(sitemap)
    soup=BeautifulSoup(response.text,"xml")
    for url_tag in soup.find_all("url"):
        loc=url_tag.find("loc")
        if loc:
            all_url.append(loc.text)
all_url=list(set(all_url))
print(f"total url founs:{len(all_url)}")
for url in all_url[:20]:
    print(url)
df=pd.DataFrame({"url":all_url})
df.to_csv("simelabs_urls.csv",index=False)
print("\n csv saved")
# for url in all_url:
#     if "machine-learning" in url:
#         print(url)