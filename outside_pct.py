import pandas as pd
df = pd.read_csv('label_containment_report.csv')
fail_df = df[df['verdict'] == 'FAIL']
print(fail_df['outside_pct'].describe())
print('\nCases with exactly 100% outside:', (fail_df['outside_pct'] == 100.0).sum())
print('Cases with partial overlap (0% < outside < 100%):', ((fail_df['outside_pct'] > 0) & (fail_df['outside_pct'] < 100)).sum())
