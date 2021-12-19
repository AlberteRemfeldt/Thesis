import numpy as np
import pandas as pd 
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib

#Load data 
X = pd.read_csv("data/X_large.csv")
y = np.load("data/y_new.npy")

#split into test and training
(X_train, X_test, y_train, y_test) = train_test_split(X, 
                                                      y, 
                                                      test_size = 0.2, 
                                                      random_state = 42)

#load models
RPF_grid = joblib.load('Results/RPF_Interactions_L2_Xlarge.pkl')
BASE_grid = joblib.load('Results/RPF_Baseline_L2_Xlarge.pkl')
RF_grid = joblib.load('Results/RF_grid.pkl')
rpf_model = RPF_grid.best_estimator_['randomplantedforestv3']
base_model = RPF_grid.best_estimator_['randomplantedforestv3']
rf_model = RPF_grid.best_estimator_['randomforestclassifier']

#impute test data 
imputer = RF_grid.best_estimator_['imputer']
X_test_trans = imputer.transform(X_test)

# permutation importance
rpf_result = permutation_importance(rpf_model, X_test_trans, y_test, n_repeats=10, random_state=42, 
                                scoring = ['neg_log_loss', 'accuracy'])
joblib.dump(rpf_result, 'Results/RPF_Interactions_Permutations.pkl')
base_result = permutation_importance(base_model, X_test_trans, y_test, n_repeats=10, random_state=42, 
                                scoring = ['neg_log_loss', 'accuracy'])
joblib.dump(base_result, 'Results/RPF_Baseline_Permutations.pkl')
rf_result = permutation_importance(base_model, X_test_trans, y_test, n_repeats=10, random_state=42, 
                                scoring = ['neg_log_loss', 'accuracy'])
joblib.dump(rf_result, 'Results/RF_Permutations.pkl')




