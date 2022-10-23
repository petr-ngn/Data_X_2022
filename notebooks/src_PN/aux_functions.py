"""
Author: Petr Nguyen

Content: User-defined auxiliary functions used in the Data-X Mushroom Classification notebook.
"""

import re
import time
import pickle
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from optbinning import BinningProcess

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.feature_selection import RFECV
from itertools import compress
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, precision_score, f1_score, recall_score, brier_score_loss, confusion_matrix, roc_curve, accuracy_score

from scipy.stats import chi2_contingency, ks_2samp
import shap






def reading_data(file_path = None):

    """

    Function for reading the raw dataset.

    Arguments:
            file_path - by default None, if we want to upload the data straight from the GitHub repository.
                - If not None, input a string with your file path, including the path and the file name.
    
                - If you type your file path with single back slashes (\) separating the folder names, you should either:
                        (1) replace them with double back slashes (\\) or,
                        (2) with forward slashes (/) or,
                        (3) put letter r in front of the string.
        
                        More information about the solution can be found here:
                https://stackoverflow.com/questions/1347791/unicode-error-unicodeescape-codec-cant-decode-bytes-cannot-open-text-file"

    Output:
        - Data frame of raw data.
    
    """

    if file_path != None:
        data = pd.read_csv(file_path)

    else:
        data = pd.read_csv('https://raw.githubusercontent.com/petr-ngn/Data_X_2022/main/data/raw.csv')

    
    return data





def removing_duplicates(data):
    
    """

    Function for printing the number of duplicated rows, which also returns a data frame without any duplicates.

    Arguments:
        data - Data frame from which we want to remove the duplicates.

    Output:
        Data frame with omitted duplicates.

    """

    print(data.duplicated().sum(), 'duplicated rows.')
    
    return data[~data.duplicated()]





def reading_cats_names_kaggle(file_path = None):

    """

    Function for extracting and displaying full categories' names and their initials per each variable.
        - This information was explicitly extracted from Kaggle and then uploaded to GitHub.

    Arguments:
        file_path - by default None, if we want to upload the data straight from the GitHub repository.
                - If not None, input a string with your file path, including the path and the file name.
    
                - If you type your file path with single back slashes (\) separating the folder names, you should either:
                    (1) replace them with double back slashes (\\) or,
                    (2) with forward slashes (/) or,
                    (3) put letter r in front of the string.

                    More information about the solution can be found here:
            https://stackoverflow.com/questions/1347791/unicode-error-unicodeescape-codec-cant-decode-bytes-cannot-open-text-file"

    Output:
        Data frame which contains the feature names, the initials of its categories and the full names of its categories.
        
    """

    if file_path != None:
        with open (file_path, 'r') as f:
            contents = f.read()
    
    else:
        #Extracting the text from the txt file.
        url = "https://raw.githubusercontent.com/petr-ngn/Data_X_2022/main/data/category_names_kaggle.txt"
        page = requests.get(url)
        contents = page.text
    
    col_dict = {}

    #For each split row (where row has following format ... variable: full category name=initials of category name):
    for c in contents.split('\n'):
        cats_dict = {}
        #Extracting column name.
        col = c.split(':')[0]
        #Extracting the categories' full names and their initials..
        cats_temp = c.split(':')[1].split(',')
        cats = []

        #For each pair full name category=category initials, split the full name and the initials from the strings.
        for cc in cats_temp:
            cats.append(cc.strip())

        #Separate both categories' full names and initials and assign to the new variables.
        cats_short = [i.split('=')[1] for i in cats]
        cats_full = [i.split('=')[0] for i in cats]

        #Storing the full name underneath the initial name:
        for k,v in zip(cats_short, cats_full):
            cats_dict[k] = v

        #Storing the full name-initials dictionary underneath the column dictionary.
        col_dict[col] = cats_dict

    df_col_desc = pd.DataFrame(columns = ['Initials','Full name'])

    #Storing the full names, initials with the variable names as row indices into a dataframe.
    for k in col_dict.keys():
        temp_df = pd.DataFrame(pd.Series(col_dict[k],index=col_dict[k].keys()),
                                columns = ['Full name']).reset_index().rename(columns = {'index':'Initials'})
                                
        temp_df = temp_df.set_index(pd.Index([k for i in range(temp_df.shape[0])]))
        df_col_desc = pd.concat((df_col_desc, temp_df))

    return df_col_desc





def mapping_cat_names(raw_data, cat_names_df):

    """
    
    Function for mapping the full categories' names to the initials. (for better readability of both data and plots)

    Arguments:
        raw_data - Data frame of raw data which contains initials of categories (which we want to replace with the full name of categories).
        cat_names_df - Data frame which contains the feature names, the initials of its categories and the full names of its categories
                        - It is the output from the function reading_cats_names_kaggle()
        
    Output:
        Data frame of the input data which contains full categories' names.

    """

    data = raw_data.copy()

    for col in list(data.columns):
   
        data[col] = data[col].replace(list(cat_names_df.loc[col,'Initials']),
                                    list(cat_names_df.loc[col,'Full name']))
                                    
    return data





def feat_dist_plot(data, conditional = False):
    
    """

    Function for plotting the distribution of features.
        - It can either display unconditional distribution or a distribution conditional on the target variable.

    Arguments:
        data - Dataframe which contains data about the features (and about the target optionally).
        conditional - True if we want to depict the features distribution conditional on the target variable, otherwise False.
    Output:
        Plot containing (conditional) distribution of each feature.
        
    """
    
    fig, axs = plt.subplots(nrows = 8, ncols = 3, figsize = (15, 35))

    for i, ax in zip(data.columns[data.columns != 'class'], axs.ravel()):

        sns.countplot(data = data, x = i ,ax = ax, hue = 'class' if conditional == True else None, palette = "ch:.25")

        ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 13)
        ax.set_title(i, fontsize = 15)
        ax.set(xlabel = None)
        
        if conditional:
            ax.legend(loc = 'best')

    #Deleting the empty plots.
    for j in range(1, 3):
        fig.delaxes(axs[7][j])

    fig.tight_layout()
    plt.show()





def dependency_analysis(data, var1, var2 = 'class', alpha = 0.05):

    
    """

    This function performs dependency analysis using chi-squared statistics. (Created by Lukas Dolezal, adjusted by Petr Nguyen)
        - H0 = On given confidence level, the occurrence of outcomes is statistically independent.
        - H1 = Non H0

    Arguments:
        data - Data frame containing data about the variables (on which the chi-squared hypothesis test will be applied).
        var1 - string name of the column/variable.
        var2 - string name of the column/variable - by default we want to perform the test with respect to the target variable.
        alpha = Confidence level; 0.05 by default.

    Outputs:
        - Print of the p-value and result of hypothesis testing.
        - Data frame of:
            - frequency/contigency table.
            - expected values specifying what the values of each cell of frequency table would be...
                    - ... if there was no association between the two variables.
    
    """


    data_crosstab = pd.crosstab(data[var1], data[var2], margins = False).rename(columns = {'edible': 'observed values - edible',
                                                                                            'poisonous': 'observed values - poisonous'})

    chi2, p, dof, expected = chi2_contingency(data_crosstab, correction = False)

    expected_df = pd.DataFrame(expected, index = data_crosstab.index)
    expected_df.columns = ['expected values - edible', 'expected values - poisonous']

    joined_crosstab_expected = pd.concat((data_crosstab, expected_df), axis = 1)

    print(f'The p-value is {p}.')

    if p <= alpha:
        print(f'Result: We reject the null hypothesis: There is a relationship between {var1} and {var2}, thus they are dependent.')
    else:
        print(f'Result: We fail to reject the null hypothesis: There is no relationship between {var1} and {var2}, thus they are independent.')

    return joined_crosstab_expected





def data_split(data_to_split, seed, valid = False):
    
    """

    Function for splitting the data into training, test and optionally validation set.
        - Before the split, it first binarizes the target variable and then separates the target variable and features.

    Arguments:
        data_to_split - Data Frame which contains data about both features and target.
        seed - Globally defined random seed in order to preserve the reproducibility.
        valid - Optional argument - If True, the data will be split into training, validation and test set; Otherwise into training and test set.
    
    Outputs:
        4 or 6 Data frames of features and target data split into training, (validation), test sets.

    """

    data = data_to_split.copy()

    #Mapping the 'edible' or 'e' class as 1 and 'poisonous' or 'p' as 0 otherwise.    
    data['class'] = data['class'].apply(lambda x: 1 if re.search("e", x) != None else 0)

    #Extracting features data and labels data.
    Y = data['class']
    X = data.drop('class', axis = 1)

    #Returning training, validation and test features and labels data (70%/15%/15%)
    if valid:
        X_temp, X_test, y_temp, y_test = train_test_split(X, Y, stratify = Y, test_size = 0.15, random_state = seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, stratify = y_temp, test_size = 0.1275, random_state = seed)

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    #Or returning training and test features and labels data only. (70%/30%)
    else:

        return train_test_split(X, Y, stratify = Y, test_size = 0.3, random_state = seed)





def binning(x_train_set, y_train_labels, x_holdout_set, cat_vars, save_binning_woe_model = False):

    """

    Function for binning the variables optimally with respect to the target variable
            - based on implementation of a rigorous and flexible mathematical programming formulation.
        - It is fitted on  training set, based on which it transforms both training and holdout set,
            - with binning (grouping) the categories for each variable.
        - Subsequently, all the categories are transformed to the numerical values, using the Weight-of-Evidence
            - (which is fitted from the training set as well).
        - The output also includes a data frame which stores all the information about the binned categories,
            - (such as grouped categories names, event rate, WoE etc.) for each variable.

    Arguments:
        x_train_set - Data frame of features used for training the model.
        y_train_labels - Data frame or Series of labels used for training the model.
        x_holdout_set - Data frame of features which is not used in the training the model (for instance, Validation set or Test set of features).
        cat_vars - List of the categorical features in x_train_set.
        save_binning_woe_model - True if we want to export the fitted Binning object in h5 format, otherwise False.

    Outputs (3 outputs):
        x_train_binned - Transformed training set of features with the binned categories which are transformed into WoE values.
        x_holdout_set_binned - Transformed Validation/Test set of features with the binned categories which are transformed into WoE values.
                                - (based on the fitted Binning object on training set).
        woe_bins - Data frame which contains information about binned features' categories
                    - (including WoE values, number of events/non-events, category frequencies etc.).

    """

    #Initializing the binning process object.
    bn = BinningProcess(variable_names = list(x_train_set.columns), categorical_variables = cat_vars)

    #Fitting the binning on training set.
    bn.fit(x_train_set,y_train_labels)

    #Transforming both training set and test set based on the fitted training binning.
    x_train_binned = bn.transform(x_train_set, metric='woe')
    x_train_binned.index = x_train_set.index
    x_holdout_set_binned = bn.transform(x_holdout_set, metric='woe')
    x_holdout_set_binned.index = x_holdout_set.index

    bins_woe = pd.DataFrame()
    
    #DataFrame including binned categories' information.
    for i in x_train_set.columns:
        
        var = bn.get_binned_variable(i).binning_table.build()
        var = var[(~var['Bin'].isin(['Special', 'Missing'])) & (~var.index.isin(['Totals']))]
        var['Variable'] = i

        bins_woe = pd.concat((bins_woe, var))
        
    if save_binning_woe_model:
        bn.save('binning_woe_model.h5')

        
    return x_train_binned, x_holdout_set_binned, bins_woe.loc[:,~bins_woe.columns.isin(['IV','JS'])]





def prep_data_export(features, labels, ind_sets, export = False, csvname = ''):

    """

    Function for joining the X train/validation/test sets of features and y train/validation/y test sets of labels into a single data frame.
     - Optionally, it can also export this joined data set into csv file.

    Arguments:
        features - list of data frames of features ... e.g., [X_train, X_valid, X_test]
        labels - list of series or data frames of labels ... e.g., [y_train, y_valid, y_set]
        ind_sets - list of sets names ... e.g., ['Training', 'Validation', 'Test']
                    - this will distinguish the instances in the data frame (which instance belongs to training set etc.)
        export - True if we want to export the joined data set into csv format, otherwise False
        csvname - The name of the csv file.
    
    Output:
        Data frame - which do have joined Training/Validation/Test features and labels.

    """

    df_list = []
    
    
    #Join each pair of features and labels data, assign to it a set indicator,
        # append to the list and then,
        #  transform that list into a data frame (and export it).
    for feat, lab, ind in zip(features, labels, ind_sets):
        
        temp = pd.concat((lab, feat), axis = 1)
        temp['set'] = ind
        df_list.append(temp)
    
    dfs = [df for df in df_list]

    if export:
        pd.concat(dfs, axis=0).sort_index().to_csv(csvname+'.csv', index = False)
    
    return pd.concat(dfs, axis=0).sort_index()





def drop_cols(bins_woe, *args):
    
    """
    
    Function for dropping the columns from the features dataset, which includes only one category after the grouping/binning.

    Arguments:
        bins_woe - Data frame which contains information about binned features' categories (including WoE values).
        *args - Any data frames of features (separated by commas), from which we want to remove the columns.
        
    Output:
        None - it removes the columns from the features data frames (X sets) specified within *args.

    """

    cols_drop = bins_woe['Variable'].value_counts(ascending = True)[bins_woe['Variable'].value_counts(ascending = True) == 1].index.tolist()

    for arg in args:
        arg.drop(cols_drop, axis=1, inplace = True, errors = 'ignore')





def woe_bins_plot(bins_woe,x_set):

    """

    Function for plotting the WoE values of each features' categories.
    
    Arguments:
        bins_woe - Data frame which contains information about binned features' categories (including WoE values).
    
    Output:
        Plot containing WoE distribution of features' categories.
    
    """

    fig, axs = plt.subplots(nrows = 6, ncols = 3, figsize = (15, 20))

    for i, ax in zip(list(x_set.columns), axs.ravel()):
    
        temp = bins_woe.loc[bins_woe['Variable'] == i]
        sns.barplot(x = temp.index, y = 'WoE', data = temp, ax = ax, palette = "ch:.25_r")
 
        labels = [str(k.tolist()).replace('[','').replace(']','').replace("'",'') for k in temp['Bin']]
        
        ax.set_title(i)
        ax.set_xticklabels(labels,rotation=50,size=10)

    fig.tight_layout()
    plt.show()





def cats_indicators(x_set, bins_woe, target_class = 'edible'):

    """
    Function outputs a data frame of variables' categories, which should have implied a target class (based on WoE coefficient).
    
    Arguments:
        x_set - Data frame containing features data (from this data frame, we extract the columns' names - the features' names).
        bins_woe - Data frame which contains information about binned features' categories (including WoE values).
        target_class = 'edible' or 'poisonous' - if former, then we filter categories having negative WoE and vice versa.


    Output:
        Data Frame containing Feature names and features categories which implies the target_class.

    """

    #Filter the subset of features' categories which satisfy the WoE-target_class condition.
    if target_class == 'edible':
        filtered_woe_bins = bins_woe[bins_woe['WoE']<= 0]

    elif target_class == 'poisonous':
        filtered_woe_bins = bins_woe[bins_woe['WoE'] >= 0]

    var_cats_dict = {}

    #For each feature, store all the categories into one list which will be append to the final data frame together with the feature name.
    for var in x_set.columns:

        #If the given variable does not have any categories which would satisfy the condition, then pass.
        if filtered_woe_bins.loc[filtered_woe_bins['Variable'] == var, 'Bin'].shape[0] == 0:
            pass

        else:
            cats_list = []

            for i in filtered_woe_bins.loc[filtered_woe_bins['Variable'] == var, 'Bin']:
                for j in i.tolist():
                    
                    cats_list.append(j)

            var_cats_dict[var] = ', '.join(cats_list)

    return pd.DataFrame(var_cats_dict.items(), columns=['Features', 'Categories'])





def bayesian_optimization(model, x_train, y_train, seed, max_features_constraint = False):

    """

    This function tunes hyperparameters of a model using Bayesian Optimization while maximizing objective function F1 score.
        - It is tuned on the training set.
        - It is tuned using Stratified 10-fold Cross Validation.

    Arguments:
        model - the model itself with default hyperparameters.
        x_train - Data frame which contains features used for training the model.
        y_train - Data frame (or series) which contains labels for training the model.
        seed - Globally defined random seed in order to preserve the reproducibility.
        max_features_constraint - This argument needs to bet set to True when tuning model which has hypeparameter max_features
                                - It can raise a value error in hyperparameter tuning within selection of final model,...
                                        -  ... when max_features is higher than the number of features in the sample:
                                                - "ValueError: max_features must be in (0, n_features]"
                                        - This issue can occur when model A chooses 5 features in the feature selection,..
                                                - but when tuning model B on the sample having 5 features chosen by model B,..
                                                - during tuning the max_features can exceed the number of features.
                                        - Therefore, the max_features needs to be adjusted accordingly to the number of features within given sample.
    
    Output:
        Model with the tuned hyperparameters.

    """

    estimator = model

    #Adjustment of max_features hyperparameter within the final model selection.
    max_features = 15 if max_features_constraint == False else len(x_train.columns)


    #Defining a searching space of possible ranges of hyperparameters, given the model.
    if type(model) == type(RandomForestClassifier()):

        search_space = {
                        'n_estimators': Integer(1, 1000),
                        'criterion': Categorical(['gini', 'entropy']),
                        'max_depth': Integer(1, 15),
                        'max_features': Integer(3, max_features),
                        'min_samples_leaf': Integer(5, 500)
                        }


    elif type(model) == type(GradientBoostingClassifier()):

        search_space = {
                        'n_estimators': Integer(1, 1000),
                         'max_depth': Integer(1, 15),
                         "learning_rate":Real(0.001, 100),
                         'min_samples_leaf': Integer(5, 500),
                         'max_features': Integer(3, max_features)
                        }


    elif type(model) == type(LogisticRegression()):

        search_space = {
                        'fit_intercept':  Categorical([True, False]),
                        'C': Real(0.001, 1000)
                        }


    elif type(model) == type(DecisionTreeClassifier()):
        
        search_space = {
                        'criterion': Categorical(['gini', 'entropy']),
                        'max_depth': Integer(1, 15),
                        'max_features': Integer(3, max_features),
                        'min_samples_leaf': Integer(5, 500)
                        }
    

    #Initialization of the stratified 10-fold cross validation.
    cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

    #Initialization of the Bayesian Optimization.
        # using the stratified 10-fold cross validation and given model, while maximizing the objective function F1 score.
    bayescv = BayesSearchCV(estimator = estimator,
                            search_spaces = search_space,
                            scoring = "f1", cv = cv,
                            n_jobs = -1, n_iter = 50,
                            verbose = 0, refit = True,
                            random_state = seed)

    #Fitting the Bayesian optimization algorithm on the training data.
    bayescv.fit(x_train, y_train)

    #Outputs the model with the best tuned hyperparameters with respect to F1 score.
    return bayescv.best_estimator_





def feat_selection(x_train, y_train, models_dict, seed):

    """

    This function is used for feature selection using RFE which requires some model as an input.

    This function takes each model, tunes it hyperparameters using Bayesian Optimization with 10-fold Cross Validation.
                        - (defined in the previous step - tuned on the training set).
        - Then uses this tuned model for Recursive Feature Elimination (RFE) with 10-fold Cross Validation.
                        - in order to choose the optimal set of features (performed on the training set).
        
    If we have 4 models as an input, then we get 4 different sets of optimal features.

    Arguments:
        x_train - Data frame which contains features used for training the model.
        y_train - Data frame (or series) which contains labels for training the model.
        models_dict - Dictionary where the keys are model names and the values are the model objects with default hyperparameters.
        seed - Globally defined random seed in order to preserve the reproducibility.
    
    Output:
        Data frame:
            Columns:
                model_name - The name of the model (the model within the models_dict parameter)
                model - The model itself with the tuned hyperparameters which then has been used within RFE (feature selection)
                rfe_model - RFE object itself (when applying transform() method, it should transform given X set on X set with optimal features).
                n_features - Number of features selected by RFE.
                final_features - The list of features selected by RFE.
                execution_time - How long it takes to tune model's hyperparameters and choose the optimal features (in seconds).

    """
    
    #Empty list for storing all the models with their tuned hyperparameters, number of features selected, and names of selected features.
    models_feats_list = []

    #For each model, tune its hyperparameters using Bayesian Optimization.
        # Then use the tuned model for feature selection using RFE while maximizing an objective function F1 Score.
    for name, mod in models_dict.items():
        print(f'Starting Feature Selection with {name} ...', '\n')

        start = time.time()
        print(f'Starting Bayesian Optimization of {name} ...')

        opt_mod = bayesian_optimization(mod, x_train, y_train, seed)

        print(f'Bayesian Optimization of {name} finished  ...')

        #Initialization of the stratified 10-fold cross validation.
        cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)

        #adjustment of the min_features_to_select parameter within the RFE based on the tuned max_features hyperparameter
            # otherwise it would raise an error.
        if any("features" in s for s in opt_mod.get_params()):
            for par in list(opt_mod.get_params()):
                if re.search('features', par) != None:
                    num_feats = opt_mod.get_params()[par]
                    break
        else:
            num_feats = 1

        print(f'Starting Recursive Feature Elimination with {name} ...')

        #Initialization of the Recursive Feature Elimination 10-fold Cross Validation in order to select the optimal number of features
            # based on the model with tuned hyperparameters and maximizing F1 score function.
        rfecv = RFECV(
                    estimator = opt_mod, step = 1,
                    min_features_to_select = num_feats,
                    scoring = 'f1', cv = cv, n_jobs = -1, verbose = 0
                    )
    
        rfecv.fit(x_train, y_train)

        print(f'Recursive Feature Elimination with {name} finished ...')
        end = time.time()

        #Extracting the final selected features and number of the selected features.
        selected_feats = list(compress(x_train.columns.tolist(), rfecv.support_.tolist()))
        number_features = len(selected_feats)
        
        print('Execution time:', round(end - start), 'seconds')
        print(number_features, 'features selected:', selected_feats, '\n')
        print('-------------------------------------------------------------------------------------------------------------------', '\n')
        
        models_feats_list.append([name, rfecv.estimator_, rfecv, number_features, selected_feats, end - start])
    
    #Storing all the models with their tuned hyperparameters, number of features selected, and names of selected features.
    feat_sel = pd.DataFrame(models_feats_list, columns = ['model_name', 'model', 'rfe_model',
                                    'n_features','final_features', 'execution_time'])

    return feat_sel





def hyperparameter_tuning(x_train, y_train, x_val, y_val, models_dict, feat_sel, seed):
    
    """

    Function for hyperparameter tuning using Bayesian Optimization with 10-fold Stratified Cross Validation.

    Arguments:
        x_train - Data frame which contains features used for training the model.
        y_train - Data frame (or series) which contains labels for training the model.
        x_val - Data frame which contains features used for evaluation on validation set. (to select the final model)
        y_val -  Data frame (or series) which contains labels evaluation on validation set. (to select the final model)
        models_dict - Dictionary where the keys are model names and the values are the model objects with default hyperparameters.
        feat_sel - Data frame outputted by feat_selection() function which contains data about:
                    - each model such as model name, model itself tuned within feature selection, number and name of final features.
        seed - Globally defined random seed in order to preserve the reproducibility.

        For each model (with default hyperparameters) and for each set of selected features (by RFE),
        ... tune its hyperparameters on training set with given selected features,...
        ... and then use this optimized model for evaluation on validation set.
    Thus, each model should be fit on 4 different set of features.
        - hence for each model, we should have 4 different sets of tuned hyperparameters (4 different models), resulting in 16 tuned models.
    
    Output:
        Data frame:
            Columns:
                tuned_model_name - The name of the tuned model (the model within the models_dict parameter)
                fs_model_name - The name of the model which has been used in feature selection (which entered the RFE-CV in the previous step).
                fs_model - The model from feature selection itself (with given hyperparameters which were tuned within feature selection).
                tuned_model - The tuned model itself (the final model which will be tuned in this phase and then evaluated on validation set).
                rfe_model - RFE model itself (when applying transform() method, it should transform given X set on X set with optimal features).
                n_features - Number of features (in the training set) on which the tuned_model has been trained.
                final_features - The list of features on which the tuned_model has been trained.
                execution_time - How long it takes to tuned hyperparameters model (in seconds).
                ... Then follow the columns which are the metrics such as AUC, Gini, Brier, KS, Precision, Recall and F1.

    """

    #Metrics space.
    metrics = {
                'F1': f1_score,
                'Precision': precision_score, 
                'Recall': recall_score, 
                'Accuracy': accuracy_score,
                'AUC': roc_auc_score, 
                'Gini': roc_auc_score, 
                'KS': ks_2samp,
                'Brier': brier_score_loss
                }

    probs_evs = ['AUC','Brier']
    class_evs = ['Accuracy', 'Precision','Recall','F1']
    
    tuned_list = []
    
    #For each model, optimize it on the subset of optimal features on training set,
        # and then evaluate it on validation set with filtered features (using set of several metrics).
    for name, mod in models_dict.items():
        
        for i in range(feat_sel.shape[0]):

            fs_name, fs_mod, rfecv = feat_sel.loc[i, 'model_name'], feat_sel.loc[i, 'model'], feat_sel.loc[i, 'rfe_model']
            final_features = [feat for feat in feat_sel.loc[i, 'final_features']]

            X_train_filtered = x_train[final_features]
            X_val_filtered = x_val[final_features]

            print(f'Starting Bayesian Optimization of {name} with {len(final_features)} features selected by {fs_name} ...', '\n')

            start = time.time()
            opt_mod = bayesian_optimization(mod, X_train_filtered, y_train, seed, max_features_constraint = True)
            end = time.time()

            evs_list = []
            
            for metric in metrics.keys():
                if metric in probs_evs:
                    evs_list.append(metrics[metric](y_val, opt_mod.predict_proba(X_val_filtered)[:, 1]))
                elif metric in class_evs:
                    evs_list.append(metrics[metric](y_val, opt_mod.predict(X_val_filtered)))
                elif metric == 'Gini':
                    evs_list.append(2* metrics['AUC'](y_val, opt_mod.predict_proba(X_val_filtered)[:, 1]) - 1)
                elif metric == 'KS':
                    X_Y_concat = pd.concat((y_val,X_val_filtered), axis = 1)
                    X_Y_concat['prob'] =  opt_mod.predict_proba(X_val_filtered)[:, 1]
                    evs_list.append(ks_2samp(X_Y_concat.loc[X_Y_concat['class'] == 1, 'prob'],
                                            X_Y_concat.loc[X_Y_concat['class'] == 0, 'prob'])[0])


            print(f'Bayesian Optimization finished ...')

            tuned_list.append([name, fs_name, fs_mod, opt_mod, rfecv,
                                len(final_features), final_features, end - start]+evs_list)
            
            print(f'Tuned hyperparameters of {name}:', opt_mod.get_params())
            print('F1 Score on Validation set:', evs_list[-1])
            print('Execution time:', round(end - start), 'seconds')
            print('-------------------------------------------------------------------------------------------------------------------', '\n')


    hyp_df = pd.DataFrame(tuned_list,
                columns = ['tuned_model_name', 'fs_model_name', 'fs_model','tuned_model',
                                'rfe_model', 'n_features', 'final_features', 'execution_time']
                                +list(metrics.keys())
                            ).sort_values(

                                        ['F1', 'Precision', 'Recall', 'Accuracy',
                                        'AUC', 'Gini', 'KS', 'Brier'],

                                        ascending = [False, False, False, False,
                                                    False, False, False, True]

                                        ).reset_index(drop = True)

    return hyp_df





def data_filter_join(hyp_tuning_df, x_train, x_valid, x_test, y_train, y_valid, model_order = 0):

    """

    Function which joins the training and validation sets of features and labels into one set.
        - then filters only the relevant final features only (the latter also applies to the test set as well).
 
    Arguments:
        hyp_tuning_df - Data frame outputted by hyperparameter_tuning() function
                      - contains data about each tuned model such as the model name, model itself, number and name of the final features, scores.
        x_train - Data frame of training set of features data.
        x_valid - Data frame of validation set of features data.
        x_test - Data frame of test set of features data.
        y_train - Data frame/Series of true labels from training set.
        y_valid - Data frame/Series of true labels from validation set.
        model_order - by default 0, is we want to select the highest ranked model (which has row index 0 in hyp_tuning_df)
        
    Outputs:
        y_train_valid - Data frame/Series with joined true labels from training and validation sets.
        x_train_valid_filtered - Data frame with joined features' data from training and validation sets, containing final features only.
        x_test_filtered - Data frame with features' data from test set, containing final features only.
        
    """
    
    final_features = [feat for feat in hyp_tuning_df.loc[model_order, 'final_features']]

    y_train_valid = pd.concat((y_train, y_valid))
    x_train_valid_filtered = pd.concat((x_train, x_valid))[final_features]
    x_test_filtered = x_test[final_features]

    return y_train_valid, x_train_valid_filtered, x_test_filtered





def final_model_fit(x_fit, y_fit, hyp_tuning_df, model_order = 0, save_models = [False, False, False]):

    """
    Function for building/fitting the final, tuned, chosen model on the joined training and validation sets.
        - Optionally, it can either export final model, feature selection and/or RFE object which are related to the final model.
    
    Arguments:
        x_fit - Data frame containing features data, on which the data will be trained.
        y_fit - Data frame/Series containing label data, on which the data will be trained.
        hyp_tuning_df - Data frame outputted by hyperparameter_tuning() function.
                      - contains data about each tuned model such as the model name, model itself, number and name of the final features, scores.
        model_order - by default 0, is we want to select the highest ranked model (which has row index 0 in hyp_tuning_df)
        save_models - list of 3 Booleans which indicates whether to save given models in the h5 format.
            - 1st Boolean for saving the tuned feature selection model.
            - 2nd Boolean for saving the fit RFE model.
            - 3rd Boolean for saving the final, tuned model.
        
    Output:
        - fitted model

    """
    
    final_model = hyp_tuning_df.loc[model_order,"tuned_model"]

    final_model.fit(x_fit, y_fit)

    if save_models[0]:
        fs_model = hyp_tuning_df.loc[model_order, 'fs_model']
        fs_model_file = open("feature_selection_model.h5", "wb")
        pickle.dump(fs_model, fs_model_file)
        fs_model_file.close()

    if save_models[1]:
        rfe_model = hyp_tuning_df.loc[model_order, "rfe_model"]
        rfe_model_file = open("rfe_model.h5", "wb")
        pickle.dump(rfe_model, rfe_model_file)
        rfe_model_file.close()

        
    if save_models[2]:
        final_model = hyp_tuning_df.loc[model_order, 'tuned_model']
        final_model_file = open("final_model.h5", "wb")
        pickle.dump(final_model, final_model_file)
        final_model_file.close()

    return final_model





def conf_mat(y_actuals, model, sample):

    """

    Function for outputting the confusion matrix as a data frame.
    
    Arguments:
        y_actuals - Data frame or series of test true labels
        model - fitted model ready for prediction.
        sample - Data frame of test features.
        
    Output:
        Data frame - confusion matrix
        
    """

    confm = pd.DataFrame(confusion_matrix(y_actuals, model.predict(sample))).rename(
                                            columns = {0: 'Predicted - Poisonous',1: 'Predicted - Edible'},
                                            index = {0: 'Actual - Poisonous', 1: 'Actual - Edible'})
    return confm





def evaluation_metrics(x_set, true_labels, model, metrics_list):

    """

    Function for outputting a data frame which depicts a list of evaluation metrics and scores of given model.
        - based on the dataset on which the model is being evaluated.
    
    Arguments:
        x_set - Data frame of test data of features.
        true_labels - Data frame or Series of test data of true labels.
        model - fitted model ready for prediction.
        metrics_list - List of metrics as strings which we want to calculate.

    Output:
        Data frame with metrics' names and their values for given model.
        
    """

    metrics = {'F1': f1_score,
                'Precision': precision_score, 
                'Recall': recall_score, 
                'Accuracy': accuracy_score,
                'AUC': roc_auc_score,
                'Gini': roc_auc_score,
                'KS': ks_2samp, 
                'Brier': brier_score_loss}

    if len(list(set(metrics_list) - set(list(metrics.keys())))) !=0:
        raise ValueError('These metrics are not acceptable'.format(list(set(metrics_list) - set(list(metrics.keys())))))

    probs_evs = ['AUC','Brier']
    class_evs = ['Precision','Recall','F1']
    evs_list = []
    
    for met in metrics_list:
        if met in probs_evs:
            evs_list.append([met, metrics[met](true_labels, model.predict_proba(x_set)[:, 1])])
        elif met in class_evs:
            evs_list.append([met, metrics[met](true_labels, model.predict(x_set))])
        elif met == 'Gini':
            evs_list.append([met, 2 *  metrics[met](true_labels, model.predict_proba(x_set)[:, 1]) - 1])
        elif met == 'KS':
            X_Y_concat = pd.concat((true_labels, x_set), axis=1)
            X_Y_concat['prob'] =  model.predict_proba(x_set)[:, 1]
            evs_list.append([met,  metrics[met](X_Y_concat.loc[X_Y_concat['class'] == 1, 'prob'],
                                                X_Y_concat.loc[X_Y_concat['class'] == 0, 'prob'])[0]])

            
    return pd.DataFrame(evs_list, columns = ['Metric', 'Score'])





def ROC_curve_plot(y_test_true, x_test, model):

    """

    Function for plotting the ROC curve.

    Arguments:
        y_test_true - Data frame/Series containing labels data from test set.
        x_test - Data frame containing features data from test set.
        model - The fitted model ready for predictions.
    
    Output:
        ROC plot

    """

    y_test_scores = model.predict_proba(x_test)[:, 1]
    FPR, TPR, _ = roc_curve(y_test_true, y_test_scores)
    AUC = round(roc_auc_score(y_test_true, y_test_scores), 2)

    fig, ax = plt.subplots()
    plt.title('Receiver Operating Characteristic Curve')

    ax.plot(FPR, TPR, 'b', label =f"AUC = {AUC}")
    ax.plot([0, 1], [0, 1], 'r--')

    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive rate')

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    plt.grid()
    plt.legend()
    plt.show()





def learning_curve_plot(model, x_set, y_set, seed):

    """

    Function for plotting the learning curve.

    Arguments:
        model - Fitted model ready for predictions.
        x_set - Data frame containing features data on which the model has been trained on.
        y_set - Data frame/series containing labels data on which the model has been trained on.
        seed - Globally defined random seed in order to preserve the reproducibility.

    Output - Plot of learning curve on training set and hold out set.

    """
    
    train_sizes, train_scores, test_scores = learning_curve(model, x_set, y_set,
                                                    cv = 10, scoring = 'f1',
                                                    train_sizes = np.linspace(0.01, 1.0, 100),
                                                    random_state = seed)

    train_mean = np.mean(train_scores, axis = 1)
    train_std = np.std(train_scores, axis = 1)
    test_mean = np.mean(test_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)

    plt.plot(train_sizes, train_mean,
                color = 'blue', marker = 'o',
                markersize = 3,
                label = 'Training F1 score')

    plt.fill_between(train_sizes,
                        train_mean + train_std,
                        train_mean - train_std,
                        alpha = 0.2, color = 'blue')

    plt.plot(train_sizes, test_mean,
                color = 'green', marker = 'o',
                linestyle = '--', markersize = 3,
                label = 'Test F1 score')

    plt.fill_between(train_sizes,
                        test_mean + test_std,
                        test_mean - test_std,
                        alpha = 0.2, color = 'green')
    
    plt.title('Learning Curve')
    plt.xlabel('Training set size')
    plt.ylabel('F1 score')

    plt.grid()
    plt.legend(loc = 'best')
    plt.show()





def shap_plots(x_set, model):

    """

    Function for plotting SHAP values.

    Arguments:
        x_set - Data frame containing set of features.
        model - fitted model ready for predictions.

    Output:
        Plot - of SHAP values per feature.

    """

    if type(model) in [type(RandomForestClassifier()), type(DecisionTreeClassifier()), type(GradientBoostingClassifier())]:
        shap_values = shap.TreeExplainer(model).shap_values(x_set)

        if type(model) == type(GradientBoostingClassifier()):
            shap.summary_plot(shap_values, x_set.values, feature_names = x_set.columns)

        elif type(model) == type(RandomForestClassifier()):
            shap.summary_plot(shap_values[1], x_set.values, feature_names = x_set.columns)

    elif type(model) == type(LogisticRegression()):
        shap.summary_plot(shap.LinearExplainer(model,
                                        shap.maskers.Independent(data = x_set)).shap_values(x_set),
                                        x_set.values, feature_names = x_set.columns)