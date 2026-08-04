import glob
import os

import pandas as pd

print('--- DATASET YEAR RANGES ---')

# 1. COT
cot_files = glob.glob('data/raw/cot/fut_fin_txt_*.zip')
if cot_files:
    years = [int(f.split('_')[-1].split('.')[0]) for f in cot_files if 'txt_' in f]
    if years:
        print(f'COT Data: {min(years)} to {max(years)}')
else:
    print('COT Data: None found')

# 2. Eco Calendar
calendar_file = 'data/raw/eco_calendar/events.csv'
if os.path.exists(calendar_file):
    try:
        df = pd.read_csv(calendar_file, usecols=['date'])
        df['date'] = pd.to_datetime(df['date'])
        min_y = df['date'].dt.year.min()
        max_y = df['date'].dt.year.max()
        print(f'ForexFactory Calendar: {min_y} to {max_y}')
    except Exception as e:
        print(f'ForexFactory Calendar: Error parsing - {e}')
else:
    print('ForexFactory Calendar: None found')

# 3. News Data
news_dir = 'data/raw/news/'
if os.path.exists(news_dir):
    try:
        print('News Data: Checking parquet files...')
        for pf in glob.glob(os.path.join(news_dir, '*.parquet')):
            print(f'  - {os.path.basename(pf)}')
    except Exception as e:
        print(f'News Data: Error parsing - {e}')

# 4. Dukascopy
dukascopy_dir = 'data/raw/dukascopy'
if os.path.exists(dukascopy_dir):
    years = set()
    for root, dirs, files in os.walk(dukascopy_dir):
        for d in dirs:
            if len(d) == 4 and d.isdigit():
                years.add(int(d))
    if years:
        print(f'Dukascopy Price Data: {min(years)} to {max(years)}')
    else:
        print('Dukascopy Price Data: No year folders found (might be still downloading)')
