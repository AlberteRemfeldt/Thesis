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

#Extracting transformer and estimator
imputer = grid_Base.best_estimator_['imputer']
rpf_int = grid_Int.best_estimator_['randomplantedforestv3']
rpf_base = grid_Base.best_estimator_['randomplantedforestv3']

#Transforming data
X_train_trans = imputer.transform(X_train)
X_test_trans = imputer.transform(X_test)

#Purifying interactions model
rpf_int.purify(X_train_trans)
joblib.dump(rpf_int, 'Results/Purified_RPF_Interactions_L2_Xlarge.pkl')


#Purifying baseline model
rpf_base.purify(X_train_trans)
joblib.dump(rpf_base, 'Results/Purified_RPF_Baseline_L2_Xlarge.pkl')



#Predictions with interpretations from interaction model
train_pur_proba_int = rpf_int.predict_purified(X_train_trans)
test_pur_proba_int = rpf_int.predict_purified(X_test_trans)
train_pur_proba_int.to_pickle('Results/train_pur_proba_L2_int.pkl')
test_pur_proba_int.to_pickle('Results/test_pur_proba_L2_int.pkl')

#Predictions with interpretations from baseline model
train_pur_proba_base = rpf_base.predict_purified(X_train_trans)
test_pur_proba_base = rpf_base.predict_purified(X_test_trans)
train_pur_proba_base.to_pickle('Results/train_pur_proba_L2_base.pkl')
test_pur_proba_base.to_pickle('Results/test_pur_proba_L2_base.pkl')


