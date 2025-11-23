import pandas as pd 
import requests 
from pathlib import Path 
from functions import get_data
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time 
''' script for downloading images from Galaxy Zoo 2 sdss dr7. 

downloads 15,000 jpeg images from the dataset, 

this may take some time.

api details can be found at the following link. 
https://skyserver.sdss.org/dr7/en/help/docs/api.asp?utm_source=chatgpt.com
'''

# directory to image folder 
dir = Path('/home/benr/ACT/CW2/sdss_images')
#create folder if it doesn't already exist 
dir.mkdir(parents=True, exist_ok=True)
df = get_data()
#ids to find galaxies 
RA_COL, DEC_COL, ID_COL = 'ra', 'dec', 'dr7objid'
df_clean = df.dropna(subset=[RA_COL,DEC_COL,ID_COL])
#use only 15,000 samples 
df_subset = df_clean.sample(n=15000,random_state=11)
#api url 
BASE_URL = 'https://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx'

# keeps a single connection open for all requests, much faster. 
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
)
session.mount("https://", adapter)
session.mount("http://", adapter)

# finds image of specific galaxy from gz2_hart16.csv.gz 
def download_image(row):
    ra = row[RA_COL]
    dec = row[DEC_COL]
    id = str(row[ID_COL])
    # api parameters 
    params = {'ra': ra, 'dec': dec, 'scale':0.4,
              'width': 256,'height':256}
    file_name = dir /f'{id}.jpg'
    if file_name.exists():
        return True
    attempts = 3
    # attempt download 3 times in case of errors 
    for retry in range(attempts):
        try:
            r = session.get(BASE_URL,params=params,timeout=20)
        #handle request errors    
        except requests.RequestException as e:
            print(f"Request failed for {id}: {e}")
            return False
        status = r.status_code
        #sleep if rate error occurs 
        if status == 429 or (500 <= status < 600):
            if retry < attempts - 1:
                time.sleep(0.5)
                continue
            else:
                print(f"Giving up on {id} with status {status}")
                return False

        # stop if any other status code given  
        if status != 200:
            print(f"HTTP {status} for {id}")
            return False

        # check to make sure file contains an image 
        ctype = r.headers.get("Content-Type", "").lower()
        if "image" not in ctype:
            print(f"Non-image content for {id}: {ctype}")
            # Optional: print a preview to see what's going on
            # print(r.text[:200])
            return False
       # save image to sdss_images folder 
        with open(file_name, 'wb') as f:
            f.write(r.content)
        time.sleep(0.8)
        return True 
# convert to list of rows so we can pass to executor
rows = list(df_subset.to_dict("records"))

# 3 workers increases download speed 
max_workers = 3

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(download_image, rows), total=len(rows)))





