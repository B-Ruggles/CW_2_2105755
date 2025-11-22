import pandas as pd 
import requests 
from pathlib import Path 
from functions import get_data
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time 
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


session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
)
session.mount("https://", adapter)
session.mount("http://", adapter)
def download_image(row):
    ra = row[RA_COL]
    dec = row[DEC_COL]
    id = str(row[ID_COL])
    params = {'ra': ra, 'dec': dec, 'scale':0.4,
              'width': 256,'height':256}
    file_name = dir /f'{id}.jpg'
    if file_name.exists():
        return True
    attempts = 3
    for retry in range(attempts):
        try:
            r = session.get(BASE_URL,params=params,timeout=20)
           
        except requests.RequestException as e:
            print(f"Request failed for {id}: {e}")
            return False
        status = r.status_code
        if status == 429 or (500 <= status < 600):
            if retry < attempts - 1:
                time.sleep(0.5)
                continue
            else:
                print(f"Giving up on {id} with status {status}")
                return False

        # For anything that's not 200, give up
        if status != 200:
            print(f"HTTP {status} for {id}")
            return False

        # Check content type is actually an image
        ctype = r.headers.get("Content-Type", "").lower()
        if "image" not in ctype:
            print(f"Non-image content for {id}: {ctype}")
            # Optional: print a preview to see what's going on
            # print(r.text[:200])
            return False
       
        with open(file_name, 'wb') as f:
            f.write(r.content)
        time.sleep(0.8)
        return True 
# convert to list of rows so we can pass to executor
rows = list(df_subset.to_dict("records"))

# adjust max_workers if you want more/less parallelism
max_workers = 3

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(tqdm(executor.map(download_image, rows), total=len(rows)))





