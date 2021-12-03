# Required Imports
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# ML Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

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
###########################################################################################################################################

def train_linear_regression(X_trn, y_trn, X_tst, y_tst):
    """" 
    This function is to train a logistic regression model using grid search, train a dummy model for that type, and plot some results
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
    Returns: the best estimator from the grid search 
    """
    # Test a few different hyperparameters for Linear Regression: n_jobs, fit_intercept
    n_jobs_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    intercept_vals = [True, False]
    param_grid_lr ={'n_jobs': n_jobs_vals, 'fit_intercept':intercept_vals}

    # Train a dummy Linear Regression model with default values
    dummy_lr = train_dummy_model(X_trn, y_trn, X_tst, y_tst, sklearn.linear_model.LinearRegression)

    # Train different Linear regression models, using grid search and cross validation to find best hyperparameters.
    gs_lr = grid_search(X_trn, y_trn, X_tst, y_tst, sklearn.linear_model.LinearRegression,param_grid_lr)

    # Plot heatmap of the gridsearch
    plot_gridsearch_2(gs_lr, n_jobs_vals, intercept_vals, 'n_jobs', 'fit_intercept')
    
    return gs_lr.best_estimator_


def train_svm_regression(X_trn, y_trn, X_tst, y_tst):
    """" 
    This function is to train a SVM model using grid search, train a dummy model for that type, and plot some results
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
    Returns: the best estimator from the grid search 
    """
    # Test a few different hyperparameters for SVM: C, gamma
    c_vals_svm = [0.01,0.1,1.0,10.0,100.0,1000.0]
    g_vals_svm = [0.001,0.01,0.1,1.0,10.0]
    param_grid_svm = {'C': c_vals_svm, 'gamma' : g_vals_svm}

    # Train a dummy SVM model with default values
    dummy_svm = train_dummy_model(X_trn, y_trn, X_tst, y_tst, sklearn.svm.SVR)

    # Train different svm models, using grid search and cross validation to find best hyperparameters.
    gs_svm = grid_search(X_trn, y_trn, X_tst, y_tst, sklearn.svm.SVR,param_grid_svm)

    # Plot heatmap of the gridsearch
    plot_gridsearch_2(gs_svm, c_vals_svm, g_vals_svm, 'C', 'gamma')

    return gs_svm.best_estimator_



def train_decision_tree_regression(X_trn, y_trn, X_tst, y_tst):
    """" 
    This function is to train a decision tree model using grid search, train a dummy model for that type, and plot some results
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
    Returns: the best estimator from the grid search 
    """
    # Test a few different hyperparameters for Decision Tree: max_depth, min_samples_leaf
    dep_vals_dt = [5, 10, 25, 50, 100]
    samp_leaf_dt = [5, 10, 25, 50, 100]
    param_grid_dt = {'max_depth' : dep_vals_dt, 'min_samples_leaf' : samp_leaf_dt}

    # Train a dummy Decision Tree model with default values
    dummy_dt = train_dummy_model(X_trn, y_trn, X_tst, y_tst, sklearn.tree.DecisionTreeRegressor)

    # Train different decision tree  models, using grid search and cross validation to find best hyperparameters.
    gs_dt = grid_search(X_trn, y_trn, X_tst, y_tst, sklearn.tree.DecisionTreeRegressor,param_grid_dt,random_state=0)

    # Plot heatmap of the gridsearch
    plot_gridsearch_2(gs_dt, dep_vals_dt, samp_leaf_dt, 'max_depth', 'min_samples_leaf')
 
    return gs_dt.best_estimator_


def train_random_forest_regression(X_trn, y_trn, X_tst, y_tst):
    """" 
    This function is to train a random forest model using grid search, train a dummy model for that type, and plot some results
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
    Returns: the best estimator from the grid search 
    """
    # Test a few different hyperparameters for Random Forest: max_depth, n_estimators
    dep_vals_rf = [10,25,50, 100,250,500,1000]
    est_vals_rf = [1,5,25,50,100,250,500,1000]
    param_grid_rf = {'max_depth': dep_vals_rf, 'n_estimators' : est_vals_rf}

    # Train a dummy Random Forest model with default values
    dummy_rf = train_dummy_model(X_trn, y_trn, X_tst, y_tst, sklearn.ensemble.RandomForestRegressor)

    # Train different random forest models, using grid search and cross validation to find best hyperparameters.
    gs_rf = grid_search(X_trn, y_trn, X_tst, y_tst, sklearn.ensemble.RandomForestRegressor,param_grid_rf,random_state=0)

    # Plot heatmap of the gridsearch
    plot_gridsearch_2(gs_rf, dep_vals_rf, est_vals_rf, 'max_depth', 'n_estimators')

    return gs_rf.best_estimator_


def train_knn_regression(X_trn, y_trn, X_tst, y_tst):
    """" 
    This function is to train a k-nearest neighbors model using grid search, train a dummy model for that type, and plot some results
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
    Returns: the best estimator from the grid search 
    """
    # Test a few different hyperparameters for K-nearest neighbours: n_neighbors, algorithm
    n_vals_k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    m_vals_k = ['euclidean', 'manhattan', 'minkowski']
    param_grid_knn = {'n_neighbors': n_vals_k, 'metric' : m_vals_k}

    # Train a dummy KNN model with default values
    dummy_knn = train_dummy_model(X_trn, y_trn, X_tst, y_tst, sklearn.neighbors.KNeighborsRegressor)

    # Train different KNN models, using grid search and cross validation to find best hyperparameters.
    gs_knn = grid_search(X_trn, y_trn, X_tst, y_tst, sklearn.neighbors.KNeighborsRegressor,param_grid_knn)

    # Plot heatmap of the gridsearch
    plot_gridsearch_2(gs_knn, n_vals_k, m_vals_k, 'n_neighbors', 'algorithm')

    return gs_knn.best_estimator_


def train_ada_regression(X_trn, y_trn, X_tst, y_tst):
    """" 
    This function is to train a AdaBoost model using grid search, train a dummy model for that type, and plot some results
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
    Returns: the best estimator from the grid search 
    """
    # Test a few different hyperparameters for AdaBoost: n_estimators, learning_rate
    est_vals_ada = [1,5,10,25,50,80,85,100]
    learn_vals_ada = [0.1, 0.5, 1.0, 1.5, 2.0]
    param_grid_ada = {'n_estimators': est_vals_ada, 'learning_rate' : learn_vals_ada}

    # Train a dummy AdaBoost model with default values
    dummy_ada = train_dummy_model(X_trn, y_trn, X_tst, y_tst, sklearn.ensemble.AdaBoostRegressor)

    # Train different AdaBoost models, using grid search and cross validation to find best hyperparameters.
    gs_ada = grid_search(X_trn, y_trn, X_tst, y_tst, sklearn.ensemble.AdaBoostRegressor,param_grid_ada, random_state=0)

    # Plot heatmap of the gridsearch
    plot_gridsearch_2(gs_ada, est_vals_ada, learn_vals_ada, 'n_estimators', 'learning_rate')

    return gs_ada.best_estimator_


def train_gaussian_process_regression(X_trn, y_trn, X_tst, y_tst):
    """" 
    This function is to train a gaussian naive bayes model using grid search, train a dummy model for that type, and plot some results
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
    Returns: the best estimator from the grid search 
    """
    # Test a few different hyperparameters for Gaussian process regression: alpha
    alpha_gpr = [0.05, 0.1, 0.2, 0.3]
    param_grid_gnb = {'alpha' : alpha_gpr}

    # Train a dummy Gaussian process regression model with default values
    dummy_gpr = train_dummy_model(X_trn, y_trn, X_tst, y_tst, sklearn.gaussian_process.GaussianProcessRegressor)

    # Train different Gaussian process regression models, using grid search and cross validation to find best hyperparameters.
    gs_gpr = grid_search(X_trn, y_trn, X_tst, y_tst, sklearn.gaussian_process.GaussianProcessRegressor,param_grid_gnb)

    # Plot heatmap of the gridsearch
    plot_gridsearch_1(gs_gpr, alpha_gpr, 'alpha')

    return gs_gnb.best_estimator_



def train_neural_network_regression(X_trn, y_trn, X_tst, y_tst):
    """" 
    This function is to train a neural network model using grid search, train a dummy model for that type, and plot some results
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
    Returns: the best estimator from the grid search 
    """
    # Test a few different hyperparameters for Neural Network Regression: hidden_layer_sizes, solver
    hl_vals = [(),(10,),(50,),(100,),(10,10),(50,50),(100,50),(100,100)]
    solv_vals = ['sgd','adam']
    param_grid_nnr = {'hidden_layer_sizes' : hl_vals, 'solver' : solv_vals}

    # Train a dummy Neural Net model with default values
    dummy_nn = train_dummy_model(X_trn, y_trn, X_tst, y_tst, sklearn.neural_network.MLPRegressor)

    # Train different Neural Net models, using grid search and cross validation to find best hyperparameters.
    gs_nn = grid_search(X_trn, y_trn, X_tst, y_tst, sklearn.neural_network.MLPRegressor,param_grid_nnr,activation='relu',max_iter=1000,batch_size=100,learning_rate_init=0.01,random_state=0)

    # Plot heatmap of the gridsearch
    plot_gridsearch_2(gs_nn, hl_vals, solv_vals, 'hidden_layer_sizes', 'solver')

    return gs_nn.best_estimator_



def train_all_classifiers(X_trn,y_trn,X_tst,y_tst):
    """" 
    This function is to call the 8 functions to train all the different types of classifiers on a dataset
    Parameters:
        X_trn - the features from training set
        y_trn - the targets from training set 
        X_tst - the features from testing set
        y_tst - the targets from testing set 
    """
    train_logistic_classifier(X_trn,y_trn,X_tst,y_tst)
    train_svm_classifier(X_trn,y_trn,X_tst,y_tst)
    train_decision_tree_classifier(X_trn,y_trn,X_tst,y_tst)
    train_random_forest_classifier(X_trn,y_trn,X_tst,y_tst)
    train_knn_classifier(X_trn,y_trn,X_tst,y_tst)
    train_ada_classifier(X_trn,y_trn,X_tst,y_tst)
    train_naive_bayes_classifier(X_trn,y_trn,X_tst,y_tst)
    train_neural_network_classifier(X_trn,y_trn,X_tst,y_tst)