#load main packages and estoimator
import pandas as pd
import numpy as np
import joblib
from datetime import date, datetime


#Load data 
X = pd.read_csv("data/X_large.csv")
y = np.load("data/y_new.npy")

from sklearn.model_selection import train_test_split

#split into test and training
(X_train, X_test, y_train, y_test) = train_test_split(X, 
                                                      y, 
                                                      test_size = 0.2, 
                                                      random_state = 42)


#load grids
grid_Int = joblib.load('Results/RPF_Interactions_L2_Xlarge.pkl')
grid_Base = joblib.load('Results/RPF_Baseline_L2_Xlarge.pkl')


#Predicting on training data
train_proba_int = grid_Int.best_estimator_.predict_proba(X_train)
train_proba_base = grid_Base.best_estimator_.predict_proba(X_train)
np.save('Results/train_proba_L2_int.npy',train_proba_int)
np.save('Results/train_proba_L2_base.npy',train_proba_base)


#Predicting on test data
test_proba_int = grid_Int.best_estimator_.predict_proba(X_test)
test_proba_base = grid_Base.best_estimator_.predict_proba(X_test)
np.save('Results/test_proba_L2_int.npy',test_proba_int)
np.save('Results/test_proba_L2_base.npy',test_proba_base)

