import cPickle as pickle
import datetime
from os.path import join
import os
import time
import sys
import hickle as hkl
import numpy as np
from pyearth import Earth
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

DATA_DIR = '/home/smile/edzhou/Thesis/data'
y = hkl.load(join(DATA_DIR, 'hkl_train/train_y_gzip.hkl'))
print 'Loaded y'

N = len(y)
train_proportion = 0.8
t = 7
X_t = hkl.load(
    join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
y_t = y[t:, ]

train_ind = np.array(xrange(0, int(train_proportion * N) - t))
test_ind = np.array(xrange(int(train_proportion * N) - t, len(y_t)))

X_train = X_t[train_ind, :]
y_train_mat = y_t[train_ind, :]
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test = scaler.transform(X_t[test_ind, :])
y_test_mat = y_t[test_ind, :]
y_pred_mat = np.zeros(y_test_mat.shape)
print 'Split and scaled data'

# Hyperparameters used in benchmarks.  Later should tune.
hp = dict(
        max_degree=1, # No interactions
        minspan=1,
        endspan=1,
        max_terms=500,
        allow_linear=False, # Unsure what this does
        use_fast=True,
        fast_K=20, # Higher = slower but better model
        fast_h=1
)

# Create this iteration's folder
folder_name = 'MARS_' + datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss")
folder_path = join(DATA_DIR, 'models', folder_name)
os.makedirs(folder_path)
# Dump the hyperparameter dictionary
with open(join(folder_path, 'hyperparameters.pkl'), 'wb') as f:
    pickle.dump(hp, f, -1)

training_times = []
for a in xrange(0, y.shape[1]):
    start = time.time()
    y_train = y_train_mat[:, a:(a + 1)].ravel()
    model = Earth(**hp)
    model.fit(X_train_scaled, y_train)
    end = time.time()
    print 'Fast MARS t-7, a{0} took {1} to train'.format(a, end - start)
    training_times.append(end - start)

    with open(join(DATA_DIR,
                   'models/{0}/MARS_a{1}.pkl'.format(folder_name, a)),
              'wb') as f:
        pickle.dump(model, f, -1)

    start = time.time()
    y_pred_mat[:, a] = model.predict(X_test)
    end = time.time()
    print 'Fast MARS t-7, a{0} took {1} to predict'.format(a, end - start)
    sys.stdout.flush()
RMSE = mean_squared_error(y_test_mat, y_pred_mat) ** 0.5
print RMSE
with open(join(folder_path, 'stats.txt'), 'wb') as f:
    f.write('{0}\n{1}\n'.format(str(RMSE), sum(training_times)))
