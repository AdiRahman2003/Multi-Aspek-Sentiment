import pandas as pd
import json
import sys
sys.path.insert(0, '.')

from app import get_top_aspects, normalize_aspect

# Cek top 10 aspects
top_10 = get_top_aspects()
print("Top 10 Aspects (after normalization):")
for i, (aspect, count) in enumerate(top_10.items(), 1):
    print(f"{i}. {aspect}: {count}")
    
print("\n\nTop 10 Aspects JSON:")
print(json.dumps(top_10, indent=2, ensure_ascii=False))

# Cek raw data
df = pd.read_excel('data/hasil_skenario3.xlsx')
print("\n\nRaw Aspek counts (top 15):")
raw_counts = df['Aspek'].value_counts()
print(raw_counts.head(15))
