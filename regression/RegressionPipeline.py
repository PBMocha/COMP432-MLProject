from PipelineFunctionsRegression import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
import os
plt.rcParams.update({'figure.max_open_warning': 0})

# 1.0 Wine Quality Red Dataset

# Load the data and split into features and targets, X and y
rd = pd.read_csv("./data/winequality-red.csv", delimiter = ';')
red_data = rd.to_numpy()
X = red_data[:,:-1]
y = red_data[:,-1:].astype(np.int32).reshape(1599,)

X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_regressors(X_trn,y_trn,X_tst,y_tst)

# 1.1 Wine Quality White Dataset

# Load the data and split into features and targets, X and y
rd = pd.read_csv("./data/winequality-white.csv", delimiter = ';')
red_data = rd.to_numpy()
X = red_data[:,:-1]
y = red_data[:,-1:].astype(np.int32).reshape(4898,)

X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_regressors(X_trn,y_trn,X_tst,y_tst)

# 2.0 Communities and Crime Dataset

# Load the data and split into features and targets, X and y
data = np.loadtxt("./data/communities.data", usecols=np.r_[0:1, 4:30,31:101,127:128], delimiter = ',')
X = data[:,:-1]
y = data[:,-1:].astype(np.float64).reshape(1994,)

X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_regressors(X_trn,y_trn,X_tst,y_tst)


# 3.0 QSAR aquatic toxicity Dataset

# Load the data and split into features and targets, X and y
df = pd.read_csv("./data/qsar_aquatic_toxicity.csv", delimiter = ';')
data = df.to_numpy()
X = np.delete(data,7,1)
y = data[:,7:8].astype(np.int32).reshape(545,)

X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_regressors(X_trn,y_trn,X_tst,y_tst)


# 4.0 Facebook Metrics Dataset

# Load the data and split into features and targets, X and y
fb = pd.read_csv("./data/dataset_Facebook.csv", delimiter = ';')
fb_data = fb.to_numpy()

# Assign 'Photo' a value 1, and 'Status' a value 0
for i in range(500):
    if fb_data[i][1] == 'Photo':
        fb_data[i][1] = 1
    else: 
        fb_data[i][1] = 0
        
fb_data = np.delete(fb_data,np.r_[0:1,3:7,15:18],axis=1)
X = fb_data[:,1:]
y = fb_data[:,0:1].astype(np.int32).reshape(500,)

X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_regressors(X_trn,y_trn,X_tst,y_tst)


# 5.0 Bike Sharing Dataset

# Load the data and split into features and targets, X and y
df_bike = pd.read_csv('./data/hour.csv', sep=',')
# Convert date time to integer
df_bike['dteday'] = pd.to_datetime(df_bike['dteday']).apply(lambda x : x.toordinal())
sl = slice(1000)
df_bike = (df_bike[sl])
data = df_bike.to_numpy()
X = data[:,:-1]
y = data[:,-1]

X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_regressors(X_trn,y_trn,X_tst,y_tst)


# 6.0 Student Performance Dataset

# Load the data and split into features and targets, X and y
df = pd.read_csv("./data/student-por.csv", sep = ';')
col_objs = df.select_dtypes(['object']).columns
encoder = LabelEncoder()

for col in col_objs:
    df[col] = encoder.fit_transform(df[col])

data = df.to_numpy()
X = data[:,:-1]
y = data[:,-1]

X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_regressors(X_trn,y_trn,X_tst,y_tst)


# 7.0 Concrete Compressive Strength Dataset

# Load the data and split into features and targets, X and y
data = load_xls_file(filename = './data/Concrete_Data.xls', skip_rows=1,skip_cols=0)
X, y = split_data_regression(data,8,1)

X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_regressors(X_trn,y_trn,X_tst,y_tst)


# 8.0 SGEMM GPU kernel performance Dataset

# Load the data and split into features and targets, X and y
data_sgemm = np.loadtxt('./data/sgemm_product.csv',dtype='float64',delimiter=',',skiprows=1)
sl = slice(1000)
data_sgemm = (data_sgemm[sl])
X = data_sgemm[:,:14]
# Since we have 4 target columns for 4 different runs, we compute the average of those 4 columns to use as our target
y_temp = data_sgemm[:,14:18]
y = np.average(y_temp, axis=1)

X_trn, X_tst, y_trn, y_tst = preprocess(X,y)

train_all_regressors(X_trn,y_trn,X_tst,y_tst)