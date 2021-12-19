#load main packages and estoimator
import pandas as pd
import numpy as np
import joblib
from RPF_EstimatorV3 import RandomPlantedForestV3
from Impute_Transformer import Imputer


#Load data 
X = pd.read_csv("data/X_large.csv")
y = np.load("data/y_new.npy")

#Load functions for modelling
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer



#split into test and training
(X_train, X_test, y_train, y_test) = train_test_split(X, 
                                                      y, 
                                                      test_size = 0.2, 
                                                      random_state = 42)


#define lists with categorical column names (for imputer) and indicies (for rpf)
cat_cols = ['sex_isFemale', 'physical_activity', 'platelets_isNormal', 'platelets_isIncreased',
            'platelets_isDecreased', 'urine_albumin_isNegative','urine_albumin_is>=30', 'urine_albumin_is>=100', 
            'urine_albumin_is>=300', 'urine_albumin_is>=1000', 'urine_albumin_isTrace', 
            'urine_glucose_isNegative', 'urine_glucose_isLight', 'urine_glucose_isMedium',
            'urine_glucose_isDark', 'urine_glucose_isVerydark','urine_glucose_isTrace', 
            'urine_hematest_isNegative', 'urine_hematest_isSmall', 'urine_hematest_isModerate', 
            'urine_hematest_isLarge','urine_hematest_isVerylarge', 'urine_hematest_isTrace']
imp_cat_cols = ['physical_activity', 'platelets_isNormal', 'platelets_isIncreased','platelets_isDecreased', 
                'urine_albumin_isNegative','urine_albumin_is>=30', 'urine_albumin_is>=100', 
                'urine_albumin_is>=300', 'urine_albumin_is>=1000', 'urine_albumin_isTrace', 
                'urine_glucose_isNegative', 'urine_glucose_isLight', 
                'urine_glucose_isMedium','urine_glucose_isDark', 
                'urine_glucose_isVerydark','urine_glucose_isTrace', 'urine_hematest_isNegative', 
                'urine_hematest_isSmall', 'urine_hematest_isModerate', 
                'urine_hematest_isLarge','urine_hematest_isVerylarge', 'urine_hematest_isTrace']
#indicies for rpd estimator after imputatation (categoricals will be placed last)
cat_col_inidicies = []
for i in cat_cols:
    arg = np.argwhere(X_train.columns == i)[0][0]
    cat_col_inidicies.append(arg)



#define column transformer to make imputation
imputer = Imputer(imp_cat_cols)
#define cv splits
skf = StratifiedKFold(n_splits = 5, random_state=42, shuffle=True)




## INTERACTION MODEL
#define rpf model with interactions
rpf = RandomPlantedForestV3(categorical_variables=cat_col_inidicies, cores = 64, loss = 'L2')
#define rpf pipe
pipe_rpf = make_pipeline(imputer, rpf)
#parameter search for model with interactions
param_search_rpf = {
    'randomplantedforestv3__max_interaction': Integer(2, 10),
    'randomplantedforestv3__n_trees': Integer(20, 50),
    'randomplantedforestv3__split_try': Integer(5, 15),
    'randomplantedforestv3__splits': Integer(30, 60),
    'randomplantedforestv3__t_try': Real(0.1, 0.4)
}
#define rpf grid
grid_rpf = BayesSearchCV(pipe_rpf, 
                         search_spaces = param_search_rpf,
                         n_iter = 50,
                         optimizer_kwargs = {'acq_func': 'EI'},
                         scoring = 'neg_log_loss',
                         n_jobs = -1,
                         verbose = 1,
                         refit = True,
                         cv = skf, 
                         random_state = 42)

#tune model with interactions
grid_rpf.fit(X_train, y_train)
#save search
joblib.dump(grid_rpf, 'Results/RPF_Interactions_L2_Xlarge.pkl')




## BASELINE MODEL
#define baseline rpf model
rpf_base = RandomPlantedForestV3(categorical_variables=cat_col_inidicies, cores = 64,
                                max_interaction = 1, loss = 'L2')
#define baseline rpf pipe
pipe_base = make_pipeline(imputer, rpf_base)
#parameter search for baseline model
param_search_base = {
    'randomplantedforestv3__n_trees': Integer(20, 50),
    'randomplantedforestv3__split_try': Integer(5, 15),
    'randomplantedforestv3__splits': Integer(30, 60),
    'randomplantedforestv3__t_try': Real(0.1, 0.4)
}
#define baseline grid
grid_base = BayesSearchCV(pipe_base,
                          search_spaces = param_search_base,
                          n_iter = 50,
                          optimizer_kwargs = {'acq_func': 'EI'},
                          scoring = 'neg_log_loss',
                          n_jobs = -1,
                          verbose = 1,
                          refit = True,
                          cv = skf, 
                          random_state = 42)

#tune baseline model
grid_base.fit(X_train, y_train)
#save search
joblib.dump(grid_base, 'Results/RPF_Baseline_L2_Xlarge.pkl')
