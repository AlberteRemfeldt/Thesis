import numpy as np
import pandas as pd 
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from datetime import datetime
import joblib
import shap

#Load data 
X = pd.read_csv("data/X_large.csv")
y = np.load("data/y_new.npy")

#split into test and training
(X_train, X_test, y_train, y_test) = train_test_split(X, 
                                                      y, 
                                                      test_size = 0.2, 
                                                      random_state = 42)


RF_grid = joblib.load('Results/RF_grid.pkl')
imputer = RF_grid.best_estimator_['imputer']
rf_model = RF_grid.best_estimator_['randomforestclassifier']


#Creating explainer
explainer = shap.TreeExplainer(rf_model)
X_train_trans = imputer.transform(X_train)
rows = np.random.choice(X_train_trans.shape[0], 1000, replace = False)
np.save('Results/SHAP_rand_rows.npy', rows)

#Shap computations
rf_shap = explainer(X_train_trans[rows,:])
joblib.dump(rf_shap, 'Results/RF_SHAP.pkl')

#Shap interaction value computations
rf_shap_interaction_vals = explainer.shap_interaction_values(X_train_trans[rows,:])
joblib.dump(rf_shap_interaction_vals, 'Results/RF_SHAP_interaction_values.pkl')


#Shap value computations
rf_shap_vals = explainer.shap_values(X_train_trans[rows,:])
joblib.dump(rf_shap_vals, 'Results/RF_SHAP_values.pkl')