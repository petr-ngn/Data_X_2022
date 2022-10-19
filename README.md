# Data-X Project 2022 - Mushroom classification

Repository for the project within the course **Data-X – Applied Data Analytics Models in Real World Tasks (4IT439)**, at Faculty of Informatics and Statistics, Prague University of Economics and Business (VSE). Such project is conducted using Python.

Authors: Petr Nguyen, Lukas Dolezal, Roman Pavlata, Patrik Korsch, Daniel Bihany

Deadline: 24/10/2022

We have been provided with a mushroom dataset, based on which we have to create a model which will classify whether or not is particular mushroom edible.

Such tasks include data understanding and exploration, data preprocessing with feature engineering feature selection, building a model including hyperparameter tuning, with subsequent evaluation using metrics such as precision, recall and F1 score.

## Repository structure and description
```
├── data		<- Data in raw form, transformed or data from third party sourcerss.
│    │
│    ├── category_names_kaggle.txt    <- Category names for mapping values (extracted from Kaggle).
│    ├── interim.csv			  <- Data after Optimal Binning and WoE tranformation.
│    ├── preprocessed.csv      	  <- Final preprocessed data after Binning, WoE transformation and feature selection.
│    ├── raw.csv		      	  <- The original, immutable data dump.
│
├── models		<- models which have been trained/fitted within out project (including objects for feature transformation and selection).
│    │
│    ├── binning_woe_model.h5         <- Optimal Binning and WoE transformation object (for feature preprocessing).
│    ├── feature_selection_model.h5   <- Optimized model used within feature selection (RFE).
│    ├── final_model.h5               <- The final optimized model trained on joined training and validation set.
│    ├── rfe_model.h5                 <- RFE object for feature selection.
│
├── notebooks     <- All the Python scripts and Jupyter notebooks used in the project. 
│    │
│    ├── src_PN   <- Module containing auxiliary functions.
│    │     │
│    │     ├── __init__.py            <- __init__ for treating src_PN as a module.
│    │     ├── aux_functions.py       <- Auxiliary functions used solely in the notebook.
│    │
│    ├── final_notebook_aux_functions_imported.ipynb <- Final notebook to which the auxiliary functions are imported as module.
│    ├── final_notebook_aux_functions_included.ipynb <- Final notebook including the auxiliary functions in it.
│
├── README.md                         <- The top-level README for readers using this project.
├── Report					  <- Documentation of our project within the course submission.
├── requirements.txt                  <- requirements file for reproducing the project.
```

### Loading a model
Use pickle package for loading the model to your workingspace.
``` bash
import pickle
model_loaded = pickle.load(open('model.h5', 'rb'))
```





## Machine Learning Flowchart
TBD

## Data Preprocessing
TBD (Binning + WoE)

## Feature selection
TBD (Bayesian Optimization + RFE; fit on training set)

## Final Model Selection
TBD (Bayesian Optimization; fit on training set with selected features by RFE; evaluation on validation set)

## Evaluation
TBD (evaluation on test set; confusion matrix, F1/Recall/Precision/Accuracy/AUC/Gini/Brier, Kolmogorov-Smirnov, ROC Curve, Learning Curve, SHAP values).

