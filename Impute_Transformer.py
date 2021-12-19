import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import bisect
from statistics import mode


class Imputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, categorical_features = [None], age_upper = [35,45,55,65,75]):
        self.categorical_features = categorical_features
        self.age_upper = age_upper
    
    
    def fit(self, X, y = None):
        # get rows and cols 
        nr, nc = X.shape
        self.nc = nc
        # get order of column names
        col_order = X.columns.tolist()
        self.col_order = col_order
        # note age and sex for all observations in X
        age = X['age'].to_numpy()
        sex = X['sex_isFemale'].to_numpy()
        
        # get columns to impute only
        col_imp = X.drop(['age', 'sex_isFemale'], axis = 1).columns.tolist()
        self.col_imp = col_imp
        
        #get age group for all obs
        age_grp = []
        for a in age:
            grp = bisect.bisect_left(self.age_upper, a)
            age_grp.append(grp)
        age_grp = np.array(age_grp)
        
        #data frame to collect imputation values of groups
        vals_imp = pd.DataFrame(np.arange(0,2), columns=['sex']).merge(
            pd.DataFrame(np.arange(0,len(self.age_upper)), columns = ['age']), how  = 'cross').set_index(['sex', 'age'])
        #fill out df
        for c in col_imp:
            temp_col = X[c].to_numpy()
            temp_col_df = pd.DataFrame({'age': age_grp,'sex': sex, c: temp_col})
            temp_col_df = temp_col_df[~temp_col_df.isnull().any(1)]
            if c in self.categorical_features:
                #mode for categorical variables
                vals = temp_col_df.groupby(['sex', 'age']).agg(mode)
            else:
                #median for numerical
                vals = temp_col_df.groupby(['sex', 'age']).agg('median')
            vals_imp = vals_imp.merge(vals, how = 'left', right_index=True, left_index=True)
        
        
        if sum(vals_imp.isnull().any()) > 0:
            raise ValueError("Imputation values containing NaN's")
        self.vals_imp = vals_imp.reset_index()
        
        
        return self
    
    def transform(self, X, y = None):
        nr, nc = X.shape
        cols = X.columns.tolist()
        #raise error if number or order of columns is not correct
        if not nc == self.nc:
            raise ValueError('Number of columns in X is not the same as the X the transformer was fitted to.')
        if not cols == self.col_order:
            raise ValueError('Order of columns in X does not correspond to the X the transformer was fitted to.')
        
        #copy X for transformation
        X_ = X.copy()
        # note age and sex for all observations in X
        age = X_['age'].to_numpy()
        sex = X_['sex_isFemale'].to_numpy()
        #get age group for all obs
        age_grp = []
        for a in age:
            grp = bisect.bisect_left(self.age_upper, a)
            age_grp.append(grp)
        age_grp = np.array(age_grp)
        
        for a_grp in range(len(self.age_upper)):
            # get imputation values for males in a_grp
            male_imp = self.vals_imp.loc[(self.vals_imp.age == a_grp) & 
                                        (self.vals_imp.sex == 0),:].drop(['sex', 'age'], axis = 1).to_dict('records')[0]
            # impute
            X_.loc[(age_grp == a_grp) & (sex == 0),self.col_imp] = X_.loc[(age_grp == a_grp) & (sex == 0),self.col_imp].fillna(male_imp)
            # get imputation values for females in a_grp
            female_imp = self.vals_imp.loc[(self.vals_imp.age == a_grp) & 
                                        (self.vals_imp.sex == 1),:].drop(['sex', 'age'], axis = 1).to_dict('records')[0]
            # impute
            X_.loc[(age_grp == a_grp) & (sex == 1),self.col_imp] = X_.loc[(age_grp == a_grp) & (sex == 1),self.col_imp].fillna(female_imp)
        
        return X_