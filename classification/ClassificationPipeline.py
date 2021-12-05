from PipelineFunctionsClassification import *
import os
plt.rcParams.update({'figure.max_open_warning': 0})

# 1.0 Diabetic Retinopathy Dataset

print("\n\nDIABETIC RETINOPATHY DATASET")
diabetic = os.path.join(os.path.dirname(__file__), './data/messidor_features.arff')
data_arff = load_arff_file(diabetic,data_type='float64',delimiter_used=',',skip_rows=24,use_cols=None)
X,y = split_data(data_arff,19,1)
X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_classifiers(X_trn,y_trn,X_tst,y_tst)


# 2.0 Default Credit Dataset

print("\n\nDEFAULT CREDIT DATASET")
credit = os.path.join(os.path.dirname(__file__), './data/default of credit card clients.xls')
data_xls = load_xls_file(credit,2,1)
X,y = split_data(data_xls,23,1)
y = y.astype(np.int32)
X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_classifiers(X_trn,y_trn,X_tst,y_tst)


# 3.0 Breast Cancer Dataset

print("\n\nBREAST CANCER")
# Load the data and split into features and targets, X and y
# This data is not loaded using a predefined function since it contains strings as targets, so it is dealt with differently
breast_cancer = os.path.join(os.path.dirname(__file__), './data/wdbc.data')
cols = np.arange(2,32)
X = np.loadtxt(breast_cancer,usecols=cols,delimiter=',') 
y = np.loadtxt(breast_cancer,dtype=str,usecols=1,delimiter=',')

# Assign 'M' (Malignant) a value 1, and 'B' (Benign) a value 0
for i in range(len(y)):
    if y[i] == 'M':
        y[i] = 1
    else: 
        y[i] = 0
        
X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_classifiers(X_trn,y_trn,X_tst,y_tst)


# 4.0 Statlog German Credit Dataset

print("\n\nGERMAN CREDIT DATASET")
# Load the data and split into features and targets, X and y
german = os.path.join(os.path.dirname(__file__), './data/german.data-numeric')
data = np.loadtxt(german,dtype='float64')
X, y = split_data(data,24,1)
X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_classifiers(X_trn,y_trn,X_tst,y_tst)


# 5.0 Adult Dataset

print("\n\nADULT DATASET")
adult = os.path.join(os.path.dirname(__file__), './data/adult.data')
df_adult = pd.read_csv(adult, names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship','race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'salary'])
df_adult_test = pd.read_csv(adult,skiprows=1, names=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship','race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'salary'])

# Drop education column since its the same data as education_num
df_adult_clean = df_adult.drop('education', axis=1)
df_adult_test_clean = df_adult_test.drop('education', axis=1)

cat_col = df_adult_clean.select_dtypes(['object']).columns
cat_col_test = df_adult_test_clean.select_dtypes(['object']).columns

df_adult_clean[cat_col] = df_adult_clean[cat_col].apply(lambda x: pd.factorize(x)[0])
df_adult_test_clean[cat_col_test] = df_adult_test_clean[cat_col_test].apply(lambda x: pd.factorize(x)[0])

data_trn = df_adult_clean.to_numpy()
data_tst = df_adult_test_clean.to_numpy()

# We don't preprocess the data for this data set because:
    # 1 - The dataset comes with a test set so no need to train_test_split
    # 2 - Most of the features are categorical instead of numerical, and thus don't require scaling
X_trn, y_trn = split_data(data_trn,13,target_location=1)
X_tst, y_tst = split_data(data_tst,13,target_location=1)

train_all_classifiers(X_trn,y_trn,X_tst,y_tst)


# 6.0 Yeast Dataset

print("\n\nYEAST DATASET")
yeast = os.path.join(os.path.dirname(__file__), './data/yeast.data')
data_yeast = np.loadtxt(yeast,dtype=str)
X = data_yeast[:,1:-1]
lb_encoder = LabelEncoder()
y = lb_encoder.fit_transform(data_yeast[:,-1])
X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_classifiers(X_trn,y_trn,X_tst,y_tst)


# 7.0 Thoracic Surgery Dataset

print("\n\nTHORACIC SURGERY DATASET")
thoracic = os.path.join(os.path.dirname(__file__), './data/ThoraricSurgery.arff')
data = np.loadtxt(thoracic,dtype=str,skiprows=21, delimiter = ',')
# Account for the categorical data in columns 0,3,9
column_trans = make_column_transformer((OneHotEncoder(),[0,3,9]),remainder='passthrough')
data = column_trans.fit_transform(data)
X, y = split_data(data,16,target_location=1)
# Assign 'T' (True) a value 1, and 'F' (False) a value 0
for i in range(len(y)):
    if y[i] == 'T':
        y[i] = 1
    else: 
        y[i] = 0
X = X.astype(np.float64)
y = y.astype(np.int32)
X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_classifiers(X_trn, y_trn, X_tst, y_tst)


# 8.0 Seismic Bumps Dataset

print("\n\nSEISMIC BUMPS DATASET")
seismic = os.path.join(os.path.dirname(__file__), './data/seismic-bumps.arff')
data = np.loadtxt(seismic,dtype=str,delimiter=',', skiprows=154) 
# Account for the categorical data in columns 0,1,2,7
column_trans = make_column_transformer((OneHotEncoder(),[0,1,2,7]),remainder='passthrough')
data = column_trans.fit_transform(data)
X, y = split_data(data,24,target_location=1)
X = X.astype(np.float64)
y = y.astype(np.int32)
X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_classifiers(X_trn,y_trn,X_tst,y_tst)


# Novelty Component - Zoo Animal Classification Dataset

print("\n\nZOO ANIMALS DATASET")
zoo = os.path.join(os.path.dirname(__file__), './data/zoo.csv')
cols = np.arange(1,18)
data_zoo = np.loadtxt(zoo,dtype='float64',delimiter=',',usecols=cols,skiprows=1)
X,y = split_data(data_zoo,16,1)
X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_classifiers(X_trn,y_trn,X_tst,y_tst)