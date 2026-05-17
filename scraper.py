import pandas as pd
from mlb_odds_scraper import scrape_oddsportal_mlb

data = scrape_oddsportal_mlb(2022)
if isinstance(data, list):
    df = pd.DataFrame(data)
else:
    df = data

print(f"Total games: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(df['date'].value_counts().sort_index().head(20))

