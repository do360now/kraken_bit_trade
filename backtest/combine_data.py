import pandas as pd

# Adjust paths
old_file = 'XXBTZEUR_1440.csv'          # from Kraken ZIP (up to Sep 2024)
recent_file = 'kraken_btc_eur_recent.csv'  # from above

# Load both (assume standard columns: timestamp, open, high, low, close, volume, ... )
# Kraken CSV may have Unix timestamp (seconds), adjust parsing as needed
df_old = pd.read_csv(old_file)
df_recent = pd.read_csv(recent_file)

# Standardize column names (rename if needed)
df_old = df_old.rename(columns={
    'Time': 'timestamp',  # or whatever the column is
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})
df_recent = df_recent.rename(columns={'timestamp': 'timestamp'})  # ensure consistent

# Convert timestamps to datetime
df_old['timestamp'] = pd.to_datetime(df_old['timestamp'], unit='s')  # if Unix seconds
df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'])

# Filter old to end at Sep 30, 2024 (avoid overlap)
cutoff = pd.to_datetime('2024-09-30')
df_old = df_old[df_old['timestamp'] <= cutoff]

# Concatenate, sort by time, drop duplicates
df = pd.concat([df_old, df_recent], ignore_index=True)
df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last')

# Keep only needed columns for backtester (adjust as per data_loader.py expectation)
df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# Save
df.to_csv('kraken_xxbtzeur_daily_2024-04-20_to_2026-02-17.csv', index=False)
print(f"Saved combined file with {len(df)} days")