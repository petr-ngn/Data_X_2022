# Data-X Project 2022 - Mushroom classification

Repository for the project within the course **Data-X – Applied Data Analytics Models in Real World Tasks (4IT439)**, at Faculty of Informatics and Statistics, Prague University of Economics and Business (VŠE). Such project is conducted using Python.

_**Authors:**_ Petr Nguyen, Lukas Dolezal, Roman Pavlata, Patrik Korsch, Daniel Bihany

_**Deadline:**_ 24/10/2022

We have been provided with a mushroom dataset, based on which we have to create a model which will classify whether or not is particular mushroom edible.

Such tasks include data understanding and exploration, data preprocessing with feature engineering feature selection, building a model including hyperparameter tuning, with subsequent evaluation using metrics such as precision, recall and F1 score.

## Repository structure and description
```
├── data     <- Data in raw form, transformed or data from third party sourcerss.
│    │
│    ├── category_names_kaggle.txt    <- Category names for mapping values (extracted from Kaggle).
│    ├── interim.csv                  <- Data after Optimal Binning and WoE tranformation.
│    ├── preprocessed.csv      	      <- Final preprocessed data after Binning, WoE transformation and feature selection.
│    ├── raw.csv                      <- The original, immutable data dump.
│
├── models   <- models which have been trained/fitted within out project (including objects for feature transformation and selection).
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
├── Report                            <- Documentation of our project within the course submission.
├── requirements.txt                  <- requirements file for reproducing the project.
```

### Installing the enviroment
First of all before running any scripts or notebooks, you should first install the enviroment. Ideally, you should create a new enviroment using Python version 3.9.13, since all the functions and module used in our notebooks have been running on such version.

You can create such enviroment in Anaconda, in the terminal as:
```bash
conda create -n yourenv python=3.9.13
```
Once you have created your own enviroment, you should then install all the packages with correct versions which have been used in our notebooks and on which the module has been built as well. This is being done with requirements.txt which you can find this repository. You should then download this file, open the Anaconda terminal, activate your enviroment as:

```bash
conda activate yourenv
```
And then install the enviroment:
```bash
pip install -r requirements.txt
```
Noted, if you do not change the directory in the terminal (using `cd` command), you should also add to it the path where the file is located.


### Importing the auxiliary functions
_The functions have been built with Python version 3.9.13_

Download the `src_PN` folder and the `final_notebook_aux_functions_imported.ipynb` notebook (they need to be stored in the same path/location). Or you can clone this whole repository using Git.

```
├── Your location
|       |
│       ├── src_PN
│       │      ├── __init__.py
│       │      ├── aux_functions.py 
│       │
|       ├── final_notebook_aux_functions_imported.ipynb
```

Afterwards, you will be able to import the auxiliary functions:

``` bash
import src_PN.aux_functions as aux
```
Otherwise, if you do not want to import the auxiliary functions as a module, you can download the notebook `final_notebook_aux_functions_included.ipynb`, in which the functions are defined directly.

### Loading a model
Use `pickle` package for loading the model to your working space.
``` bash
import pickle
model_loaded = pickle.load(open('model.h5', 'rb'))
```
## Machine Learning Flowchart
First, we import all the relevant data to the working space, with following data inspection and data understanding and exploration. Afterwards, the data split into training, validation and sets following with further data preprocessing in the form of binning and Weight-of-Evidence Encoding.

Once we have our data prepared, we performe an iterative process for feature selection using RFE in combination with hyperparameter tuning using Bayesian Optimization - the latter optimized the input model by tuning its hyperparameters and then this tuned model is used as an input within RFE. This results in $n$ different set of optimal features, assuming we have $n$ input models.

After the feature selection, we perform the selection of the final model. For each input model, we tune its hyperparemeters with Bayesian Optimization on each set of optimal features selected within feature selection in the previous step. Once the model is optimized, then it is evaluated on the validation set. We choose such model, which has the highest metrics score and/or lowest cost functions. This results in $n^2$ tuned models, assuming we have $n$ input models.

Once we select the best, final model which performs the best on the validation set, we then build as fitting it on the join training a and validation set. Finally, we evaluate the fitted model on the set by calculating several metrics such as F1 score, Recall, Precision, Accuracy, AUC etc.

![alt_text](https://github.com/petr-ngn/Data_X_2022/blob/main/flowchart_data_x.png?raw=true)

## Data Preprocessing
Following the data exploration, we deem reasonable to remove found duplicates (counting 6 rows) and 1 row containing errornous category _EE_ in _stalk-shape_ feature. As mentioned, we also remapped the features' categories with they full names.

After the data split into training set, validation set and test set (70%; 15%; 15%); we have to convert the categorical features into numerical ones. For this case, we use `optbinning` package, which is available [here](https://github.com/guillermo-navas-palencia/optbinning). It uses optimal binning with respect to the target variable by rigorous and flexible mathematical programming formulation - in other words, it optimally groups the features' categories into bins. It can be either used for continuous features, where instead of grouping, the data will be optimally split into interval bins.

Once it bins the categories, we decide to transform these bins into numerical values by using Weight-of-Evidence Encoding which is commonly used in credit risk modelling and works better than dummy encoding when having larger amount of variables and categories/bins.

After such transformation, we then exclude such features which do have only one bin/category after binning. These features have zero-variance, thus they would not add any contribution in terms of predictions.

## Feature selection
TBD (Bayesian Optimization + RFE; fit on training set)

## Final Model Selection
TBD (Bayesian Optimization; fit on training set with selected features by RFE; evaluation on validation set)

## Evaluation
TBD (evaluation on test set; confusion matrix, F1/Recall/Precision/Accuracy/AUC/Gini/Brier, Kolmogorov-Smirnov, ROC Curve, Learning Curve, SHAP values).

