import pandas as pd
import requests
import trafilatura
import time
url_df=pd.read_csv(r"C:\Users\vishn\data science\Simelabs\5-06-26\simelabs_urls.csv")
documents=[]
print(f"total url found:{len(url_df)}")
for url in url_df["url"]:
    try:
        response=requests.get(url,headers={
            "User-Agent":"Mozilla/5.0" },timeout=15)
        if response.status_code!=200:
            print(f"failed{url}")
            continue
        text=trafilatura.extract(response.text)
        if text:
            documents.append({
                "url":url,
                "content":text
            })
            print(f"sucessfull:{url}")
        else:
            print(f"no content{url}")
    except Exception as e:
        print(f"error:{url}")
        print(e)
    time.sleep(1)
content_df=pd.DataFrame(documents)
content_df.to_csv("simelabs_knowledge_base.csv",index=False)
print("Done")
print(f"\n document extracted{len(content_df)}")
print("doc saved")