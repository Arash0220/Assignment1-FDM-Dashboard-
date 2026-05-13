import pandas as pd
import numpy as np
import re
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

print("="*60)
print(" MOVIE DATA PREPARATION FOR POWER BI")
print("="*60)

# 1. LOAD DATA
print("\n[1/8] Loading data...")
df = pd.read_csv('movies_with_api_financials.csv')
print(f" Loaded {len(df)} rows")

# 2. STANDARDIZE TEXT COLUMNS
print("\n[2/8] Cleaning text fields...")
str_cols = ['name', 'director', 'writer', 'star', 'company', 'country', 'genre', 'rating']
for col in str_cols:
    df[col] = df[col].astype(str).str.strip().str.title()
    df[col] = df[col].replace(['Nan', 'NaN', ''], np.nan)
df['rating'] = df['rating'].str.upper().replace(['NAN', 'NOT RATED'], 'Not Rated')

# 3. CONVERT NUMERIC COLUMNS
print("\n[3/8] Converting numeric types...")
num_cols = ['year', 'score', 'votes', 'budget', 'gross', 'runtime']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. PARSE 'RELEASED' INTO DATE + COUNTRY
print("\n[4/8] Parsing release dates...")
def parse_released(text):
    if pd.isna(text) or str(text).strip() == '': return pd.NaT, np.nan
    text = str(text).strip()
    country_match = re.search(r'\(([^)]+)\)', text)
    country = country_match.group(1).strip() if country_match else np.nan
    date_str = re.sub(r'\s*\([^)]*\)', '', text).strip()
    try: return pd.to_datetime(date_str), country
    except: return pd.NaT, country

df[['release_date', 'release_country']] = df['released'].apply(lambda x: pd.Series(parse_released(x)))
df.drop('released', axis=1, inplace=True)

# 5. DROP ROWS MISSING ESSENTIAL IDENTIFIERS
print("\n[5/8] Removing invalid records...")
df = df.dropna(subset=['name', 'year']).reset_index(drop=True)
print(f" Remaining: {len(df)} rows")

# 6. CREATE TRANSPARENCY FLAGS (BEFORE IMPUTATION)- for trails
df['budget_missing_original'] = df['budget'].isna()
df['gross_missing_original'] = df['gross'].isna()
df['budget_imputed'] = False
df['gross_imputed'] = False

# 7. IMPUTE MISSING BUDGET/GROSS (GENRE+DECADE MEDIAN)
print("\n[6/8] Imputing missing financials...")
df['decade'] = (df['year'] // 10) * 10

def impute_by_group(df, target, groups):
    df = df.copy()
    df['_gkey'] = df[groups].astype(str).agg('_'.join, axis=1)
    medians = df.groupby('_gkey')[target].transform('median')
    mask = df[target].isna() & medians.notna()
    df.loc[mask, target] = medians[mask]
    df.loc[mask, f'{target}_imputed'] = True
    df.drop('_gkey', axis=1, inplace=True)
    return df

df = impute_by_group(df, 'budget', ['genre', 'decade'])
df = impute_by_group(df, 'gross', ['genre', 'decade'])

# Fallback to global median any remaining gaps (should be minimal)
for col in ['budget', 'gross']:
    if df[col].isna().any():
        med = df[col].median()
        df.loc[df[col].isna(), col] = med
        df.loc[df[col] == med, f'{col}_imputed'] = True

# Fill other gaps
df['runtime'] = df['runtime'].fillna(df['runtime'].median())
df['score'] = df['score'].fillna(df.groupby('genre')['score'].transform('mean')).fillna(df['score'].median())
cat_fill = {'rating': 'Not Rated', 'genre': 'Unknown', 'director': 'Unknown', 'writer': 'Unknown',
            'star': 'Unknown', 'country': 'Unknown', 'company': 'Independent', 'release_country': 'Unknown'}
for c, v in cat_fill.items(): df[c] = df[c].fillna(v)

# 8. calculate financial metrics (profit, ROI, classifications)
print("\n[7/8] Calculating analytical fields...")
df['profit'] = df['gross'] - df['budget']

# Raw ROI (mathematically exact)
df['roi_raw'] = np.where(df['budget'] > 0, (df['profit'] / df['budget']) * 100, np.nan)

# Display ROI (keeps outliers flagged)
df['roi_display'] = df['roi_raw'].clip(lower=-100, upper=1000)
df['roi_flagged'] = (df['roi_raw'] > 1000) | (df['roi_raw'] < -100)

df['profit_margin_pct'] = np.where(df['gross'] > 0, (df['profit'] / df['gross']) * 100, np.nan)
df['break_even_multiplier'] = np.where(df['budget'] > 0, df['gross'] / df['budget'], np.nan)

# Classification fields
def fin_success(x):
    if pd.isna(x): return 'Unknown'
    elif x >= 500: return 'Blockbuster'
    elif x >= 100: return 'Hit'
    elif x >= 0: return 'Break-Even'
    else: return 'Flop'
df['financial_success'] = df['roi_display'].apply(fin_success)

df['score'] = df['score'].clip(0, 10)
df['score_category'] = pd.cut(df['score'], bins=[-0.1, 6, 7, 8.5, 10], 
                              labels=['Poor', 'Average', 'Good', 'Excellent'])

df['runtime_category'] = pd.cut(df['runtime'], bins=[0, 90, 120, 999], 
                                labels=['Short (<90m)', 'Standard (90-120m)', 'Long (>120m)'])

df['budget_tier'] = pd.cut(df['budget'], bins=[0, 1e7, 5e7, 1e8, 999e9], 
                           labels=['Low (<$10M)', 'Medium ($10-50M)', 'High ($50-100M)', 'Blockbuster (>$100M)'])

df['decade_label'] = df['decade'].astype(str) + 's'
def era(x):
    if x < 1990: return 'Classic Era'
    elif x < 2000: return 'Modern Era'
    elif x < 2010: return 'Digital Era'
    else: return 'Streaming Era'
df['era'] = df['year'].apply(era)
df['years_since_release'] = datetime.now().year - df['year']

def success_matrix(row):
    s, r = row['score'], row['roi_display']
    if pd.isna(s) or pd.isna(r): return 'Unknown'
    elif s >= 7.5 and r >= 200: return 'Critical & Commercial Hit'
    elif s >= 7.5: return 'Critical Darling'
    elif r >= 200: return 'Commercial Hit'
    else: return 'Underperformer'
df['success_matrix'] = df.apply(success_matrix, axis=1)

df['engagement_score'] = (df['votes'] / 1000) * df['score']
df['efficiency_ratio'] = np.where(df['budget'] > 0, df['score'] / (df['budget'] / 1e6), np.nan)

# 9. FINAL TYPE CONVERSIONS FOR POWER BI
print("\n[8/8] Optimizing types & exporting...")
for c in ['year', 'decade', 'votes']: df[c] = df[c].astype('Int64')
float_cols = ['score', 'budget', 'gross', 'runtime', 'profit', 'roi_raw', 'roi_display',
              'profit_margin_pct', 'break_even_multiplier', 'engagement_score', 'efficiency_ratio']
for c in float_cols: df[c] = df[c].astype('float64')
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
for c in ['budget_missing_original', 'gross_missing_original', 'budget_imputed', 'gross_imputed', 'roi_flagged']:
    df[c] = df[c].astype('boolean')
str_final = ['name', 'rating', 'genre', 'director', 'writer', 'star', 'country', 'company',
             'release_country', 'score_category', 'runtime_category', 'budget_tier',
             'decade_label', 'era', 'financial_success', 'success_matrix']
for c in str_final: df[c] = df[c].astype(str).str.strip()

df.to_csv('movies_powerbi_ready.csv', index=False, encoding='utf-8')


print(f"• Total Movies: {len(df)}")
print(f"• Budget Coverage: {(~df['budget_missing_original']).mean()*100:.1f}%")
print(f"• Median ROI (Display): {df['roi_display'].median():.1f}%")
print(f"• Extreme ROI Flagged (for transparency): {df['roi_flagged'].sum()}")
print("\n READY FOR POWER BI. Import 'movies_powerbi_ready.csv' directly.")