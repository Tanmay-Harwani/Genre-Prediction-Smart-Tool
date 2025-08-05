import pandas as pd
import glob
import os
folder_path = '/Users/tanmay/Documents/DSP_Homework/Term Project/archive/csv/'

csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    artist_name = os.path.basename(file).replace('.csv', '')
    df['Artist'] = artist_name
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

combined_df = combined_df[['Lyric', 'Artist']].dropna()

artist_to_genre = {
    'ArianaGrande': 'Pop',
    'Beyonce': 'Pop',
    'BillieEilish': 'Alternative',
    'BTS': 'K-pop',
    'CardiB': 'Hip-Hop',
    'CharliePuth': 'Pop',
    'ColdPlay': 'Rock',
    'Drake': 'Hip-Hop',
    'DuaLipa': 'Pop',
    'EdSheeran': 'Pop',
    'Eminem': 'Rap',
    'JustinBieber': 'Pop',
    'KatyPerry': 'Pop',
    'Khalid': 'R&B',
    'LadyGaga': 'Pop',
    'Maroon5': 'Pop',
    'NickiMinaj': 'Hip-Hop',
    'PostMalone': 'Hip-Hop',
    'Rihanna': 'Pop',
    'SelenaGomez': 'Pop',
    'TaylorSwift': 'Pop'
}

combined_df['Genre'] = combined_df['Artist'].map(artist_to_genre)

combined_df = combined_df.dropna()

combined_df.to_csv('lyrics_dataset.csv', index=False)

print("✅ Combined CSVs and saved as 'lyrics_dataset.csv'")
