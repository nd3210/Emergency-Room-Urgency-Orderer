import numpy as np
import pandas as pd

df = pd.read_csv('emergency.csv')

df = df.drop(columns = ['dep_name', 'ethnicity', 'race', 'lang', 'religion', 'maritalstatus', 'employstatus', 'insurance_status', 'arrivalmode',
                        'disposition', 'previousdispo', 'arrivalmonth', 'arrivalday', 'arrivalhour_bin'])

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

test = pd.DataFrame(df.iloc[:, 6:df.columns.get_loc("cc_abdominalcramping")])

too_little = []

for i in test.columns:
    if test[i].sum() < 50:
        too_little.append(i)

for i in too_little:
    df = df[df[i] != 1]
    df = df.drop(columns = i)
    df.reset_index(drop=True)

print(df.shape)

df = df.dropna(axis=0)

print(df.shape)    

df['cc_abdominalpain'] = df['cc_abdominalcramping'] + df['cc_abdominalpain']
df['cc_breathingdifficulty'] = df['cc_breathingproblem'] + df['cc_breathingdifficulty'] + df['cc_dyspnea']
df['cc_addictionproblem'] = df['cc_alcoholproblem'] + df['cc_drugproblem']
df['cc_overdose'] = df['cc_overdose-accidental'] + df['cc_overdose-intentional']
df['cc_giproblem'] = df['cc_gibleeding'] + df['cc_giproblem']


df = df.drop(columns= ['cc_abdominalcramping', 'cc_breathingproblem', 'cc_addictionproblem', 'cc_dyspnea', 'cc_overdose-accidental', 
                       'cc_overdose-intentional'])


print(df.shape)
df.to_csv('new_emergency.csv')