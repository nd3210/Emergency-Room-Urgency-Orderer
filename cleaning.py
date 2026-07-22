import numpy as np
import pandas as pd

df = pd.read_csv('emergency.csv')

df = df.drop(columns = ['dep_name', 'ethnicity', 'race', 'lang', 'religion', 'maritalstatus', 'employstatus', 'insurance_status', 'arrivalmode',
                        'disposition', 'previousdispo'])

suffix = ('_last', '_min', '_max', '_median', '_count', '_npos')
removed_col  = [x for x in df.columns if x.endswith(suffix)]
df = df.drop(columns = removed_col)

prefix = ('meds', 'n_', 'triage_vital_')
removed_col = [x for x in df.columns if x.startswith(prefix)]
df = df.drop(columns = removed_col)

print(df.shape)

df = df.drop_duplicates()
df = df.reset_index(drop=True)
print(df.shape)

print(df.isna().sum())

df = df[(df['age'] > 0) & (df['esi'] > 0) & (df['esi'] < 6)]
df = df.reset_index(drop=True)
print(df.shape)

print(df.isna().sum())

df.to_csv('new_emergency.csv')