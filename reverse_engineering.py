import os
import pandas as pd
from collections import Counter

def rd_finder(PATH):
    minnesota_daily = [os.path.join(PATH, f) for f in os.listdir(PATH)]

    row_dict={}
    for file in minnesota_daily:
        try:
            df = pd.read_csv(file, header=None, index_col=0)
        except:
            print(file)
            continue
        try:
            row_dict[file] = list(df.loc['2022-02-12'])
        except:
            row_dict[file] = None

    return row_dict


# Create a Counter object from the values of the dictionary
#counted_values_target = Counter(tuple(v) for v in row_dict.values() if v)

rd_target=rd_finder(PATH = '84/weather/prediction_targets_daily')
rd_daily = rd_finder(PATH = '84/weather/minnesota_daily')

files = []
for k_t, v_t in rd_target.items():
    for k, v in rd_daily.items():
        if v_t == v:
            files.append([k_t,k])
a=1