# Required Imports
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# ML Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def load_arff_file(filename,data_type,delimiter_used,skip_rows,use_cols):
    """" 
    This function is to load .arff files
    Parameters:
        filename - name of the file, in our case ending with .arff
        data_type - dtype we want for the data
        delimiter_used - what delimiter the data uses
        skip_rows - number of rows to skip in the data
        use_cols - what columns of the data we want to use. Specify use_cols = None if we want all the columns
    Returns: the loaded data
    """
    data = np.loadtxt(filename,dtype=data_type,delimiter=delimiter_used,skiprows=skip_rows,usecols=use_cols) 
    return data

def load_xls_file(filename,skip_rows,skip_cols): 
    """" 
    This function is to load .xls files
    Parameters:
        filename - name of the file, in our case ending with .xls
        skip_rows - number of rows to skip in the data
        skip_cols - number of rows to skip in the data
    Returns: the loaded data
    """
    if skip_rows == 0:
        skip_rows = 1
    df = pd.read_excel(filename)
    data = df.to_numpy()
    data = data[skip_rows-1:,skip_cols:]
    return data

def split_data(data,n_features,target_location=1):
    """" 
    This function is to split an array of data into two arrays, the features (X) and the targets (y) 
    Parameters:
        data - array of data
        n_features - number of features in the data
        target_location - if the column of targets is the first or last column in the data (input 0 for first col, and 1 for last col)
    Returns: array X of features, and array y of targets
    """
    if target_location == 1:
        X = data[:,0:n_features]
        y = data[:,n_features:n_features+1].astype(np.int32)
    else:
        X = data[:,1:n_features+1]
        y = data[:,0:1].astype(np.int32)
        
    y_shape = y.shape[0]
    y = y.reshape(y_shape,)
    
    return X, y

def preprocess(X,y,test_split_size=0.3):
    """" 
    This function is to split an arrays X and y into training and testing data - X_trn, X_tst, y_trn, y_tst
    It then scales the data using StandardScaler
    Parameters:
        X - array of features
        y - array of targets
        test_split_size - ratio of (desired instances of test data / total number of data instances). Must be between 0 and 1
    Returns: Scaled data - X_trn, X_tst, y_trn, y_tst
    """
    X_trn, X_tst, y_trn, y_tst = sklearn.model_selection.train_test_split(X, y, test_size=test_split_size, random_state=0)
    scaler = sklearn.preprocessing.StandardScaler()
    X_trn = scaler.fit_transform(X_trn)
    X_tst = scaler.fit_transform(X_tst)
    return X_trn, X_tst, y_trn, y_tst

def plot_gridsearch_1(gridsearch, param_1, param_1_name,):
    """" 
    This function is to plot the mean scores of a hyperparameters of a gridsearch as a heatmap
    Parameters:
        gridsearch - the gridsearch object
        param_1 - the list of values for parameter 1
        param_1_name - a string of the name of parameter 1
    """
    scores_test = gridsearch.cv_results_['mean_test_score'].reshape(1,len(param_1))
    scores_train = gridsearch.cv_results_['mean_train_score'].reshape(1,len(param_1))
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1,2,1)
    plt.imshow(scores_train, interpolation='nearest', cmap='YlOrRd')
    plt.xlabel(param_1_name)
    plt.colorbar()
    plt.xticks(np.arange(len(param_1)), param_1)
    plt.title('Training data')
    plt.tick_params(left=False,labelleft=False)
    
    plt.subplot(1,2,2)
    plt.imshow(scores_test, interpolation='nearest', cmap='YlOrRd')
    plt.xlabel(param_1_name)
    plt.colorbar()
    plt.xticks(np.arange(len(param_1)), param_1)
    plt.title('Testing data')
    plt.tick_params(left=False,labelleft=False)
              
    plt.suptitle(f'Heatmap of {param_1_name}');
 

def plot_gridsearch_2(gridsearch, param_1, param_2, param_1_name, param_2_name):
    """" 
    This function is to plot the mean scores of a hyperparameters of a gridsearch as a heatmap
    Parameters:
        gridsearch - the gridsearch object
        param_1 - the list of values for parameter 1
        param_2 - the list of values for parameter 2
        param_1_name - a string of the name of parameter 1
        param_2_name - a string of the name of parameter 2
    """
    scores_test = gridsearch.cv_results_['mean_test_score'].reshape(len(param_2),len(param_1))
    scores_train = gridsearch.cv_results_['mean_train_score'].reshape(len(param_2),len(param_1))
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1,2,1)
    plt.imshow(scores_train, interpolation='nearest', cmap='YlOrRd')
    plt.xlabel(param_1_name)
    plt.ylabel(param_2_name)
    plt.colorbar()
    plt.xticks(np.arange(len(param_1)), param_1)
    plt.yticks(np.arange(len(param_2)), param_2)
    plt.title('Training data')
    
    plt.subplot(1,2,2)
    plt.imshow(scores_test, interpolation='nearest', cmap='YlOrRd')
    plt.xlabel(param_1_name)
    plt.ylabel(param_2_name)
    plt.colorbar()
    plt.xticks(np.arange(len(param_1)), param_1)
    plt.yticks(np.arange(len(param_2)), param_2)
    plt.title('Testing data')
              
    plt.suptitle(f'Heatmap of {param_1_name} vs. {param_2_name}');
      

def train_dummy_model(X_trn, y_trn, X_tst, y_tst,estimator_type):
    """" 
    This function is to train a model with no tuned hyperparameters
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
        estimator_type - the type of estimator to train
    Returns: the trained model
    """
    clf = estimator_type().fit(X_trn,y_trn)
    print(f'Dummy model: {clf}')
    print(f'With {round(100*clf.score(X_trn,y_trn),3)}% train accuracy') 
    print(f'With {round(100*clf.score(X_tst,y_tst),3)}% test accuracy')
    return clf

def grid_search(X_trn, y_trn, X_tst, y_tst, estimator_type, param_grid, **kwargs):     
    """" 
    This function is to train a model and perform a grid search using 3-fold cross validation
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
        estimator_type - the type of estimator to train
        param_grid - dict of parameters to use in the grid search 
        **kwargs - other parameters to use to train the model
    Returns: the GridSearchCV object
    """
    clf = estimator_type(**kwargs)
    gs_clf = sklearn.model_selection.GridSearchCV(clf,param_grid,verbose=1,cv=3,return_train_score=True).fit(X_trn,y_trn)
    print(f'Best estimator: {gs_clf.best_estimator_}')
    print(f'With {round(100*gs_clf.best_estimator_.score(X_trn,y_trn),3)}% train accuracy') 
    print(f'With {round(100*gs_clf.best_estimator_.score(X_tst,y_tst),3)}% test accuracy')
    return gs_clf