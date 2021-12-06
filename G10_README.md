# README for G10 - COMP 432 Default Project
## Comparision of various Classification and Regression ML models, as well investigating the interpretability of CNNs and DecisionTrees over the CIFAR10 dataset

### Submitted by:
#### David Rady - 40098177
#### Athigan Sinnathurai - 40132792
#### Joshua Parial-Bolusan - 40063663

### Task 1 and 2

The purpose of these tasks are to train and compare different machine learning models on datasets from the UCI repository. 

Classification Algorithms used: 

-   Logistic regression (for classification)
-   Support vector classification
-   Decision tree classification
-   Random forest classification
-   k-nearest neighbours classification
-   AdaBoost classification
-   Gaussian naive Bayes classification
-   Neural network classification

Regression Algorithms used:
-   Linear regression
-   Support vector regression
-   Decision tree regression
-   Random forest regression
-   kk-nearest neighbours regression
-   AdaBoost regression
-   Gaussian process regression
-   Neural network regression

### Task 3

The purpose of this task to compare the inerpretability of a Convolutional Neural Network using activation maximization and a Decision tree CLassifier over the CIFAR10 dataset 
Algorithsm used:
- Pytorch Convolutional Neural Network
- DecisionTreeClassifier  (sklearn)


## Dependencies (Third-party Libraries)
- numpy
- matplotlib
- pandas
- sklearn
- pytorch

### Run the following command to install all dependencies:

pip:
    
    python -m pip install numpy pandas matplotlib sklearn torch

anaconda: 

    conda install <package-name>

## File Structure
The code is organized by different tasks

- `COMP432-project`                                         -- root folder
    - `./classification/` -- contains code and jupyter notebook for running Classification models
        - `./classification/data`                        -- contains datasets that are being classified
        - `./classification/ClassificationPipeline.py`   -- contains script to run classification pipeline on all datasets
        - `./classification/ClassificationPipeline.ipynb`   -- jupyter notebook as an option to run the classification pipeline
        - `./classification/PipelineFunctionsClassification.py` -- contains helper functions for running the classifications
    - `./regression/` -- contains code and jupyter notebook for running Regression models
        - `./regression/data`                        -- contains datasets that are being used for regression
        - `./regression/PipelineFunctionsRegression.py`                       -- contains functions to help with the regression pipeline
        - `./regression/RegressionPipeline.ipynb`                       -- jupyter notebook option to run regression pipeline
        - `./regression/RegressionPipeline.py`                       -- script to run the regression pipeline on datasets
    - `./interpretability/` -- contains code and jupyter notebook for training a Convolutional Neural Network and a Decision Tree on CIFAR1- dataset
        - `./interpretability/data`                        -- contains CIFAR10 dataset
        - `./interpretability/out`                       -- contains output plots and figures from the CNN and Decision Tree
        - `./interpretability/Cifar10Classification.ipynb`  -- juyter notebook to run the interpretability task on CIFAR10
        - `./interpretability/CifarClassifers.py`                       -- contains code for handling CNN and Decision Tree classification
        - `./interpretability/CifarDataProcess.py`                       -- conatins code for handling data preprocessing on CIFAR10 
    - `./plots/` -- contains plots from the classification and regression experiments
        - `./plots/classification/`                        -- contains output plots / figures from classification models
        - `./plots/regression`                       -- contains output plots / figures from regression models
    

## How To Run
There are 2 options to run and test the program.

### Jupyter Notebook
- Open up jupyter notebook
- Navigate to the root project folder
- The tasks are sperated into different jupyter notebooks
- Run the notebook for the class you are trying to test
    - Example: ./classification/CLassificationPipelineNotebook.ipynb to run classification tasks
- For the Interpretability tasks, run the ./interpretability/CIfar10Classifcation.ipynb
### Regular Python program (only usable for classification and regression tasks)
- For classification, navigate to classification folder, then run ClassificationPipeline.py
- For regression, navigate to regression folder, then run RegressionPipeline.py
## Datasets

### UCI
- the various datasets can be downloaded here:
- Classification:
    - https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
    - https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    - https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    - https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
    - https://archive.ics.uci.edu/ml/datasets/adult
    - https://archive.ics.uci.edu/ml/datasets/Yeast
    - https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data
    - https://archive.ics.uci.edu/ml/datasets/seismic-bumps
    
- Regression:
    - http://archive.ics.uci.edu/ml/datasets/Wine+Quality
    - http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
    - http://archive.ics.uci.edu/ml/datasets/QSAR+aquatic+toxicity
    - http://archive.ics.uci.edu/ml/datasets/Facebook+metrics
    - http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    - http://archive.ics.uci.edu/ml/datasets/Student+Performance
    - http://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
    - http://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance
    
### CIFAR10 
- The CIFAR10 dataset can be downloaded here: https://www.cs.toronto.edu/~kriz/cifar.html
    - Note : Its the python version
### Kaggle
- the novelty component datasets were downloaded from kaggle:
- https://www.kaggle.com/uciml/zoo-animal-classification?select=zoo.csv
- https://www.kaggle.com/heitornunes/caffeine-content-of-drinks
