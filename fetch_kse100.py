import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import os

def fetch_kse100():
    url = 'https://en.wikipedia.org/wiki/KSE_100_Index'
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    tables = soup.find_all('table', {'class': 'wikitable'})
    target_table = None
    for t in tables:
        headers_row = [th.text.strip() for th in t.find_all('th')]
        if 'Ticker' in headers_row and 'Company' in headers_row:
            target_table = t
            break
            
    if not target_table:
        print("Could not find table")
        return
        
    data = []
    for r in target_table.find_all('tr')[1:]:
        cols = [td.text.strip() for td in r.find_all(['td', 'th'])]
        if len(cols) >= 4:
            # Ticker is in cols[0] format like 'PSX:\xa0ABOT'
            ticker_raw = cols[0]
            ticker = ticker_raw.replace('PSX:', '').replace('\xa0', '').strip()
            
            # Company
            company = cols[2]
            
            # Sector
            sector = cols[3]
            
            data.append({
                'ticker': ticker,
                'company_name': company,
                'sector': sector,
                'fetched_at': pd.Timestamp.now().strftime('%Y-%m-%d')
            })
            
    df = pd.DataFrame(data)
    
    # Save to data/kse100_constituents.csv
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/kse100_constituents.csv', index=False)
    print(f"Saved {len(df)} constituents to data/kse100_constituents.csv")
    print(df.head())

if __name__ == '__main__':
    fetch_kse100()
