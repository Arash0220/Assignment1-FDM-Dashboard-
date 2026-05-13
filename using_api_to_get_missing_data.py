# %% [markdown]
# # Step 1: Autofill Missing Budget AND Gross Values via TMDb API
# Fetches missing financial data from The Movie Database (TMDb) API
# and saves the updated dataset for further cleaning.

# %%
import pandas as pd
import numpy as np
import requests
import time
import re
from tqdm import tqdm

# %%
# Load the original dataset
df = pd.read_csv('movies.csv')
print(f"[INFO] Loaded dataset: {df.shape}")

# %%
# Identify rows with missing budget OR gross
missing_budget = df['budget'].isna() | (df['budget'].astype(str).str.strip() == '')
missing_gross = df['gross'].isna() | (df['gross'].astype(str).str.strip() == '')
missing_either = missing_budget | missing_gross

print(f"[INFO] Missing budget: {missing_budget.sum()}")
print(f"[INFO] Missing gross: {missing_gross.sum()}")
print(f"[INFO] Missing either: {missing_either.sum()}")

# %%
def get_financials_from_tmdb(movie_name, year, api_key):
    """Fetch BOTH budget and gross from TMDb API"""
    try:
        # Search for movie
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': api_key,
            'query': movie_name,
            'year': int(year) if pd.notna(year) else None
        }
        response = requests.get(search_url, params=params, timeout=10)
        results = response.json().get('results', [])
        
        if results:
            movie_id = results[0]['id']
            
            # Get detailed movie info
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            params = {'api_key': api_key}
            response = requests.get(details_url, params=params, timeout=10)
            data = response.json()
            
            budget = data.get('budget', 0)
            gross = data.get('revenue', 0)  # TMDb uses 'revenue' for box office gross
            
            return {
                'budget': budget if budget and budget > 0 else None,
                'gross': gross if gross and gross > 0 else None
            }
        return {'budget': None, 'gross': None}
    except Exception as e:
        print(f"[WARN] API error for {movie_name}: {e}")
        return {'budget': None, 'gross': None}

# %%
# API Configuration
API_KEY = "d248be4441e6855f96c5ae2cb105d4db"  # Your TMDb API key
output_file = 'movies_with_api_financials.csv'

# %%
# Fetch and fill missing financials
print("[INFO] Starting API fetch for budget AND gross...")
for idx in tqdm(df[missing_either].index, desc="Fetching financial data"):
    movie_name = str(df.loc[idx, 'name']).strip()
    year = df.loc[idx, 'year']
    
    # Clean name for API search
    clean_name = re.sub(r'[^\w\s\-]', '', movie_name)[:100]
    
    financials = get_financials_from_tmdb(clean_name, year, API_KEY)
    
    # Update budget if missing and found
    if missing_budget.iloc[idx] and financials['budget']:
        df.loc[idx, 'budget'] = financials['budget']
        print(f"[+] {movie_name} ({year}): Budget = ${financials['budget']:,}")
    
    # Update gross if missing and found
    if missing_gross.iloc[idx] and financials['gross']:
        df.loc[idx, 'gross'] = financials['gross']
        print(f"[+] {movie_name} ({year}): Gross = ${financials['gross']:,}")
    
    if not financials['budget'] and not financials['gross']:
        print(f"[-] {movie_name} ({year}): No financial data found")
    
    time.sleep(0.25)  # Respect TMDb rate limit
# %%
# Save updated dataset
df.to_csv(output_file, index=False)
print(f"[INFO] Updated dataset saved to '{output_file}'")

# %%
# Summary report
print("\n[INFO] Post-API Summary:")
print(f"- Budgets filled: {(~df['budget'].isna()).sum()} / {len(df)} ({(~df['budget'].isna()).mean()*100:.1f}%)")
print(f"- Gross values filled: {(~df['gross'].isna()).sum()} / {len(df)} ({(~df['gross'].isna()).mean()*100:.1f}%)")
print(f"- Remaining missing budgets: {df['budget'].isna().sum()}")
print(f"- Remaining missing gross: {df['gross'].isna().sum()}")