# Thesis

This repository hold files used to create the results and plots in my master's thesis.

Original data made available by the TreeEcxplainer Study repository, https://github.com/suinleelab/treeexplainer-study. 
The R files are from the Random Planted Forest repository, https://github.com/PlantedML/Planted_Forest.

## Files

Two files hold own implementations needed for compatability with scikit-learn and scikit-optimize packages:
- RPF_EstimatorV3.py hols a scikit-learn compatible estimator for the Random Planted Forest model.
- Impute_Transformer.py holds a scikit-learn compatible transformer to impute missing values using sex and age of individuals in data. 

**Files used to create results**
- **RF_pipelineV3.ipynb** performes bayesian search for Random Forest Classifier. Creates *RF_grid.pkl*.
- **RPF_L2_large.py** performes bayesian search for Baseline model and Random Planted Forest model with interactions. Note this is set up to use 64 cores. Creates *RPF_Baseline_L2_Xlarge.pkl* and *RPF_Interactions_L2_Xlarge.pkl*
- **Predict_from_models.py** predicts from opmtimal Baseline and Random Planted Forest model on both training and test data. Creates *test_proba_L2_base.npy*, *test_proba_L2_int.npy*, *train_proba_L2_base.npy* and *train_proba_L2_int.npy*.
- **Purify_models.py** purifies the opmtimal Baseline and Random Planted Forest model on both training and test data. Creates *Purified_RPF_Baseline_L2_Xlarge.pkl*, *Purified_RPF_Interactions_L2_Xlarge.pkl*, *test_pur_proba_L2_base.pkl*, *test_pur_proba_L2_int.pkl*, *train_pur_proba_L2_base.pkl*, *train_pur_proba_L2_int1.pkl* and *train_pur_proba_L2_int2.pkl*.
- **Permutation_importance.py** performs permutation feature importance on the test data from the optimal Random Forest, Baseline and Random Planted Forest models. Creates *RF_Permutations.pkl*, *RPF_Baseline_Permutations.pkl* and *RPF_Interactions_Permutations.pkl*.
- **SHAP_values.py** calculates shap, shap main and shap interaction values of the optimal Random Forest model for 1000 observations in the training data. Creates *SHAP_rand_rows.npy*, *RF_SHAP.pkl*, *RF_SHAP_values.pkl* and *RF_SHAP_interaction_values.pkl*.

**Files used to analyze results**
- **Predictive_Performance.ipynb** analysis of the predictive performance of the optimal models
- **Explanations.ipynb** analysis of the explanability of the model predictions
