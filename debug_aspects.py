import pandas as pd

df = pd.read_excel('data/hasil_skenario3.xlsx')
if 'Aspek' in df.columns:
    aspek_raw = df['Aspek'].tolist()
    print(f'Total rows: {len(aspek_raw)}')
    unique_aspeks = set(str(x) for x in aspek_raw)
    print(f'Unique raw aspects: {len(unique_aspeks)}')
    print('\nAll unique raw values:')
    for val in sorted(unique_aspeks):
        print(f'  {repr(val)}')
else:
    print('Aspek column not found')
