import pandas as pd
import sys
sys.path.insert(0, '.')
from app import normalize_aspect

df = pd.read_excel('data/hasil_skenario3.xlsx')
if 'Aspek' in df.columns:
    aspek_raw = df['Aspek'].tolist()
    normalized = [normalize_aspect(asp) for asp in aspek_raw]
    
    # Count normalized values
    from collections import Counter
    counts = Counter(normalized)
    
    print('Normalized aspect value counts:')
    for asp, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'  {repr(asp)}: {count}')
else:
    print('Aspek column not found')
