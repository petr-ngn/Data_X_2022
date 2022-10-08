"""
Author: Petr Nguyen
Collaborants: Lukas Dolezal, Patrik Korsch, Roman Pavlata, Daniel Bihany
"""


import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from optbinning import BinningProcess
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,precision_score, f1_score, recall_score, brier_score_loss, confusion_matrix
from scipy.stats import ks_2samp
import shap



def reading_data():
    """
    Function for reading the data from GitHub.
    """

    url_data = 'https://raw.githubusercontent.com/petr-ngn/Data_X_2022/main/data/mushrooms_vse.csv'
    data = pd.read_csv(url_data)

    #Converting object data types into categories for a faster comptutation.
    data = data.astype('category')
    return data



def removing_duplicates(data):
    """
    Function for printing the number of duplicated rows, which also return a data frame without any duplicates.
    """

    print(data.duplicated().sum(), 'duplicated rows.')
    
    return data[~data.duplicated()]



#Data frame for storing the full names of variables' categories.
def reading_cats_names_kaggle():
    """
    Function for extracting and displaying full categories' names instead of their initials.
    This information was explicitly extracted from Kaggle and then uploaded to GitHub.
    """

    #Extracting the text from the txt file.
    url = "https://raw.githubusercontent.com/petr-ngn/Data_X_2022/main/data/category_names_kaggle.txt"
    page = requests.get(url)
    contents = page.text
    
    col_dict = {}

    #For each split row (where row has following format ... variable: full category name=initials of category name):
    for c in contents.split('\n'):
        cats_dict = {}
        #extracting column name
        col = c.split(':')[0]
        #extracting the categories' full names and their initials.
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
    Function for mapping the full categories' names to the initials.
    """

    data = raw_data.copy()

    for col in list(data.columns):
   
        data[col] = data[col].replace(list(cat_names_df.loc[col,'Initials']),
                                    list(cat_names_df.loc[col,'Full name']))
    return data



def feat_dist_plot(data, conditional = False):
    """
    Function for plotting the distribution of features.
    It can either display unconditional distribution or a distribution conditional on the target variable.
    """
    
    fig, axs = plt.subplots(nrows = 8, ncols = 3, figsize = (15, 35))
    for i, ax in zip(data.columns[data.columns != 'class'], axs.ravel()):
        sns.countplot(data = data, x=i ,ax = ax,hue= 'class' if conditional == True else None, palette = "ch:.25")
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 13)
        ax.set_title(i, fontsize = 15)
        ax.set(xlabel = None)

    for j in range(1,3):
        fig.delaxes(axs[7][j])

    fig.tight_layout()
    plt.show()



def data_split(data_to_split, valid = False):
    """
    Function for splitting the data into training, test and optionally validation set.
    Beforand the split, it first binarizes the target variable and then separates the target variable and features.
    """
    data = data_to_split.copy()

    #Constraints check:
    if valid not in [False, True]:
        raise ValueError('Only True or False Booleans are acceptable.')
    if type(data) != type(pd.DataFrame()):
        raise ValueError('The input data has to be a Data Frame type.')
    if 'class' not in data.columns:
        raise ValueError('The target variable has to be named as "class"')

    #Mapping the 'edible' or 'e' class as 1 and 'poisonous' or 'p' as 0 otherwise.    
    data['class'] = data['class'].apply(lambda x: 1 if re.search("e", x) != None else 0)

    Y = data['class']
    X = data.drop('class', axis = 1)

    seed = 702
    if valid:
        X_temp, X_test, y_temp, y_test = train_test_split(X,Y,stratify=Y,test_size=0.3,random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_temp,y_temp,stratify=y_temp,test_size=0.3,random_state=seed)

        return X_train, X_valid, X_test, y_train, y_valid, y_train

    else:
        
        return train_test_split(X,Y,stratify=Y,test_size=0.3,random_state=seed)



def binning(x_train_set, y_train_labels, x_holdout_set, cat_vars):
    """
    Function for binning the variables optimally with respect to the target variable, based on implementation of a rigorous and flexible mathematical programming formulation.
    It is fitted on the training set, based on which it transforms both training and holdout set with binning (grouping) the categories for each variable.
    Subsequently, all the categories are transformed to the numerical values, using the Weight-of-Evidence (which is fitted from the training set as well).
    The ouput also includes a data frame which stores all the information about the binned categories (such as grouped categories names, event rate, WoE etc.) for each variable.
    """

    bn = BinningProcess(list(x_train_set.columns), categorical_variables = cat_vars)

    #Fitting the binning on training set.
    bn.fit(x_train_set,y_train_labels)

    #Transforming both training set and test set based on the fitted training binning.
    x_train_binned = bn.transform(x_train_set, metric='woe')
    x_holdout_set_binned = bn.transform(x_holdout_set, metric='woe')

    woe_bins = pd.DataFrame()
    
    #DataFrame including binned categories' information.
    for i in x_train_set.columns:
        var = bn.get_binned_variable(i).binning_table.build()
        var = var[(~var['Bin'].isin(['Special', 'Missing'])) & (~var.index.isin(['Totals']))]
        var['Variable'] = i
        woe_bins = pd.concat((woe_bins, var))

    return x_train_binned, x_holdout_set_binned, woe_bins.loc[:,~woe_bins.columns.isin(['IV','JS'])]



def drop_cols(woe_bins, *args):
    """
    Function for dropping the columns from the features dataset, which includes only one category after the grouping/binning
    """

    cols_drop = [woe_bins['Variable'].value_counts(ascending=True)[woe_bins['Variable'].value_counts(ascending=True)==1].index]
    for arg in args:
        arg.drop(cols_drop, axis=1, inplace = True, errors = 'ignore')



def conf_mat(y_actuals, model, sample):
    """
    Function for outputing the confusion matrix as a data frame
    """

    confm = pd.DataFrame(confusion_matrix(y_actuals, model.predict(sample))).rename(
                                            columns = {0:'Pred - Poisonous',1:'Pred - Edible'},
                                            index = {0:'Actual - Poisonous', 1:'Actual - Edible'})
    return confm



def evaluation_metrics_df(x_set, true_labels, model, metrics_list):
    """
    Function for outputing a data frame which depicts a list of evaluation metrics and scores of given model, based on the dataset on which the model is being evaluated.
    """

    metrics = {'AUC':roc_auc_score,
                'Gini': roc_auc_score,
                'Brier':brier_score_loss,'KS':ks_2samp,'Precision':precision_score,'Recall':recall_score,'F1':f1_score}

    if len(list(set(metrics_list)-set(list(metrics.keys())))) !=0:
        raise ValueError('These metrics are not acceptable'.format(list(set(metrics_list)-set(list(metrics.keys())))))

    probs_evs = ['AUC','Brier']
    class_evs = ['Precision','Recall','F1']
    evs_list = []
    
    for met in metrics_list:
        if met in probs_evs:
            evs_list.append([met, metrics[met](true_labels, model.predict_proba(x_set)[:,1])])
        elif met in class_evs:
            evs_list.append([met, metrics[met](true_labels, model.predict(x_set))])
        elif met == 'Gini':
            evs_list.append([met, 2*metrics['AUC'](true_labels, model.predict_proba(x_set)[:,1]) - 1])
        elif met == 'KS':
            X_Y_concat = pd.concat((true_labels.reset_index(),x_set), axis=1)
            X_Y_concat['prob'] =  model.predict_proba(x_set)[:,1]
            evs_list.append([met, ks_2samp(X_Y_concat.loc[X_Y_concat['class'] == 1,'prob'], X_Y_concat.loc[X_Y_concat['class'] == 0,'prob'])[0]])

            
    return pd.DataFrame(evs_list, columns = ['Metric','Score'])



def evaluation_models(x_set, true_labels, models_dict, metric):
    """
    Function which outputs a data frame depicting the perfomances of provided models, measured by given metric
    """

    #Dictionary of metrics and their root functions.
    metrics = {'AUC':roc_auc_score,
                'Gini': roc_auc_score,
                'Brier':brier_score_loss,'KS':ks_2samp,'Precision':precision_score,'Recall':recall_score,'F1':f1_score}

    #Constraint check
    if metric not in metrics.keys():
        raise ValueError('This metric is not acceptable.')

    probs_evs = ['AUC','Brier']
    class_evs = ['Precision','Recall','F1']
    evs_list = []

    #Calculating the metrics' scores.
    for name, mod in models_dict.items():
        if metric in probs_evs:
            evs_list.append([name, metrics[metric](true_labels, mod.predict_proba(x_set)[:,1])])
        elif metric in class_evs:
            evs_list.append([name, metrics[metric](true_labels, mod.predict(x_set))])
        elif metric == 'Gini':
            evs_list.append([name, 2*metrics['AUC'](true_labels, mod.predict_proba(x_set)[:,1]) - 1])
        elif metric == 'KS':
            X_Y_concat = pd.concat((true_labels.reset_index(),x_set), axis=1)
            X_Y_concat['prob'] =  mod.predict_proba(x_set)[:,1]
            evs_list.append([name, ks_2samp(X_Y_concat.loc[X_Y_concat['class'] == 1,'prob'], X_Y_concat.loc[X_Y_concat['class'] == 0,'prob'])[0]])

            
    return pd.DataFrame(evs_list, columns = ['Model',metric])



def shap_plots(x_set, target_class = 'edible'):
    shap_values = shap.TreeExplainer(rf).shap_values(x_set)
    shap.summary_plot(shap_values[1 if target_class == 'edible' else 0], x_set.values, feature_names = x_set.columns)



def cats_indicators(x_set, woe_bins, target_class = 'edible'):
    """
    Function outputs a data frame of variables' categories, which should have implied a target class (based on WoE coefficient).
    """

    if target_class == 'edible':
        filtered_woe_bins = woe_bins[woe_bins['WoE']<= 0]
    elif target_class == 'poisonous':
        filtered_woe_bins = woe_bins[woe_bins['WoE'] >= 0]

    var_cats_dict = {}
    for var in x_set.columns:
        if filtered_woe_bins.loc[filtered_woe_bins['Variable'] == var,'Bin'].shape[0] == 0:
            pass
        else:
            cats_list = []
            for i in filtered_woe_bins.loc[filtered_woe_bins['Variable'] == var,'Bin']:
                for j in i.tolist():
                    cats_list.append(j)
            var_cats_dict[var] = ', '.join(cats_list)

    return pd.DataFrame(var_cats_dict.items(), columns=['Features', 'Categories'])