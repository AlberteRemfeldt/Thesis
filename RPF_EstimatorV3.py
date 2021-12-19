import numpy as np
import pandas as pd
import os
import re

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import rpy2.rinterface as ri
r = ro.r

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


r['source']('rpf2.R')
rpf_func = ro.globalenv['rpf']
r['source']('predict_rpf.R')
pred_rpf = ro.globalenv['predict_rpf']
r['source']('purify_2.R')
purify = ro.globalenv['purify']
r['source']('predict_purified.R')
pred_pur = ro.globalenv['pred_pur']
r['source']('rpf_feature_importance.R')
rpf_feature_importance = ro.globalenv['rpf_feature_importance']


class RandomPlantedForestV3(BaseEstimator, ClassifierMixin):
    def __init__(self, max_interaction=2, n_trees=50, splits=30, split_try=10, t_try=0.4, variables=None, min_leaf_size=1,alternative=False,
    loss="L2", epsilon=0.1, categorical_variables=None, delta=0, cores=1, seed = 42):
        self.max_interaction=max_interaction
        self.n_trees=n_trees
        self.splits=splits
        self.split_try=split_try
        self.t_try=t_try
        self.variables=variables
        self.min_leaf_size=min_leaf_size
        self.alternative=alternative
        self.loss=loss
        self.epsilon=epsilon
        self.categorical_variables=categorical_variables
        self.delta=delta 
        self.cores=cores
        self.seed = seed
        self.pur_res_R = None


    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        #convert X and y to R objects
        ro.numpy2ri.activate()
        
        nr,nc = X.shape
        X_R = r.matrix(X, nrow=nr, ncol=nc)
        y_R = r.matrix(y, nrow = nr, ncol = 1)
        
        #convert arguments to R objects
        #if variables = None, pass NULL to R, else pass list of vectors
        if self.variables is None:
            variables_R = r('NULL')
        else:
            variables_Rindex = [x+1 for x in self.variables]
            variables_vector_R = ro.IntVector(variables_Rindex)
            r.assign("variables_vector_R", variables_vector_R)
            variables_R = r("""variables_list_R <- as.list(variables_vector_R)""")
        #make alternative into R boolean type
        alternative_R = ro.vectors.BoolVector([self.alternative])
        #make categorical_variables into R vector
        if self.categorical_variables is None:
            categorical_variables_R = r('NULL')
        else:
            categorical_variables_Rindex = [x+1 for x in self.categorical_variables]
            categorical_variables_R = ro.IntVector(categorical_variables_Rindex)

        #run RPF model as R function, returns R object
        forest_res_R = rpf_func(y_R, X_R,
                                max_interaction = self.max_interaction,
                                ntrees = self.n_trees, 
                                splits = self.splits, 
                                split_try = self.split_try, 
                                t_try = self.t_try, 
                                variables = variables_R,
                                min_leaf_size = self.min_leaf_size, 
                                alternative = alternative_R, 
                                loss = self.loss,
                                epsilon = self.epsilon, 
                                categorical_variables = categorical_variables_R,
                                delta = self.delta,   
                                cores = self.cores,
                                seed = self.seed)
        
        #save R list with forest residuals
        self.forest_res_R_ = forest_res_R

        feature_importance_R = rpf_feature_importance(forest_res_R)
        feature_importance_Py = np.asarray(feature_importance_R)
        self.feature_importance_ = feature_importance_Py
        
        ro.numpy2ri.deactivate()

        return(self)
    

    #method to get feature importance
    #def feature_importance(self):
    #    ro.numpy2ri.activate()
    #    feature_importance_R = rpf_feature_importance(res = self.forest_res_R_)
    #    feature_importance_Py = np.asarray(feature_importance_R)
    #    ro.numpy2ri.deactivate()
    #    return feature_importance_Py

    #method to purify the fitted rpf
    def purify(self, X, y = None):
        #check the estimator has been fitted 
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        nr,nc = X.shape

        #convert X to R object
        ro.numpy2ri.activate()
        X_R = r.matrix(X, nrow = nr, ncol = nc)

        #purify model
        pur_res_R = purify(self.forest_res_R_, X_R, self.cores)
        self.pur_res_R = pur_res_R
        ro.numpy2ri.deactivate()

        pur_res_PY = []
        for f in range(len(pur_res_R)):
            values = []
            for v in range(len(pur_res_R[f].rx2('values'))):
                values.append(list(pur_res_R[f].rx2('values')[v]))

            lim_list = []
            for l in range(len(pur_res_R[f].rx2('lim_list'))):
                lim_list.append(list(pur_res_R[f].rx2('lim_list')[l]))
            
            individuals = []
            for i in range(len(pur_res_R[f].rx2('individuals'))):
                if pur_res_R[f].rx2('individuals')[i] == ri.NULL:
                    individuals.append(None)
                else:
                    individuals.append(list(pur_res_R[f].rx2('individuals')[i]))

            my_functions = []
            for m in range(len(pur_res_R[f].rx2('my.functions'))):
                features = list(pur_res_R[f].rx2('my.functions')[m])
                for i in range(len(features)):
                    features[i] -= 1
                if features[0] == -1:
                    features = ['Intercept']
                my_functions.append(features)


            fam_i = {'values':values, 'lim_list': lim_list, 'individuals': individuals, 'my_functions':my_functions}

            pur_res_PY.append(fam_i)
        
        self.pur_res_PY = pur_res_PY

        return(self)


    def predict_proba(self, X, y = None):
        #check the estimator has been fitted 
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        nr,nc = X.shape

        ro.numpy2ri.activate()
        X_R = r.matrix(X, nrow = nr, ncol = nc)
        pred_R = pred_rpf(X = X_R, forest_res = self.forest_res_R_, cores = self.cores)
        pred_PY = np.asarray(pred_R)
        ro.numpy2ri.deactivate()
        if self.loss == 'logit':
            pred_PY = 1/(1+np.exp(-pred_PY))
        proba = np.vstack([1-pred_PY, pred_PY]).T
        return proba
    
    def predict(self, X, y = None):
        #check the estimator has been fitted 
        check_is_fitted(self)
        
        proba = self.predict_proba(X)
        pred = np.argmax(proba, axis = 1)
         
        return pred
    

    def predict_purified(self, X, y = None, interpret = True):
        #check the estimator has been fitted 
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        nr,nc = X.shape

        if self.pur_res_R is None:
            print('Estimator is not purified. Please purify first.')
        else:
            ro.numpy2ri.activate()
            #make alternative into R boolean type
            interpret_R = ro.vectors.BoolVector([interpret])
            X_R = r.matrix(X, nrow = nr, ncol = nc)
            pred_pur_R = pred_pur(X_R, self.pur_res_R, interpret = interpret_R, cores = self.cores)
            ro.numpy2ri.deactivate()

            pred_pur_PY = dict()
            for n in range(len(pred_pur_R.names)):
                name = pred_pur_R.names[n]
                if name == 'Y.hat':
                    name_PY = 'Pred'
                elif name == '(0)':
                    name_PY = 'Intercept'
                else:
                    name_PY = [float(x)- 1.0 for x in re.findall('[1-9][0-9]|[0-9]', name)]
                pred_pur_PY[f"{name_PY}"] = list(pred_pur_R[n])

            return pd.DataFrame(pred_pur_PY)