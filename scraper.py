import pandas as pd
import statsapi
import os
from mlb_odds_scraper import scrape_oddsportal_mlb

# ============================================================
# BASEBALL MODEL - HISTORICAL DATA SCRAPER
# Run this file separately from model.py
# Purpose: Pull and save historical odds and game logs
# for backtesting the model
# ============================================================

# Seasons to pull
SEASONS = [2021, 2022, 2023, 2024]

# Output folder for historical data
OUTPUT_FOLDER = 'historical_data'

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created folder: {OUTPUT_FOLDER}")

def pull_historical_odds(season):
    print(f"Pulling odds for {season}...")
    try:
        data = scrape_oddsportal_mlb(season)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        filename = f"{OUTPUT_FOLDER}/odds_{season}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} games to {filename}")
        return df
    except Exception as e:
        print(f"Error pulling {season}: {e}")
        return None

# Test with one year first
for season in SEASONS:
    pull_historical_odds(season)

print("\nAll done! Files saved to historical_data folder.")