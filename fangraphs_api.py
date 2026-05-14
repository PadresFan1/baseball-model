import requests
import pandas as pd
from io import StringIO

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0'
}

url = "https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitcher&year=2026&position=&team=&min=q&csv=true"

response = requests.get(url, headers=headers)
print(response.status_code)
print(response.text[:500])