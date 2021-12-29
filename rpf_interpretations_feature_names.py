import pandas as pd
import numpy as np
import re

def add_feature_names(df, col_names):
    df_new = df.copy()
    feature_names = []
    for i in range(len(list(df.columns))): 
        i_name = list(df.columns)[i]
        if (i_name == 'Intercept') | (i_name == 'Pred'):
            feature_in_col = i_name
        else:
            feature_indicies = re.findall('[0-9][0-9].0|[0-9].0', i_name)
            feature_in_col = str()
            for j in range(len(feature_indicies)):
                ind = int(float(feature_indicies[j]))
                if j == len(feature_indicies) - 1:
                    feature_in_col += col_names[ind]
                else:
                    feature_in_col += str(col_names[ind] + str('__'))
        feature_names.append(feature_in_col)
    df_new.columns = feature_names
    return df_new