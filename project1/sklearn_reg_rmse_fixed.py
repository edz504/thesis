from os.path import join
import time
import hickle as hkl
from sklearn import linear_model
import cPickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

DATA_DIR = '/home/smile/edzhou/Thesis/data'
y = hkl.load(join(DATA_DIR, 'hkl_train/train_y_gzip.hkl'))

T = 7
N = len(y)
train_proportion = 0.8

t_vec = xrange(0, 8)
RMSE_vec = [None] * len(t_vec)

for t in t_vec:
    X_t = hkl.load(
        join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
    y_t = y[t:, ]

    train_ind = np.array(xrange(0, int(train_proportion * N) - t))
    test_ind = np.array(xrange(int(train_proportion * N) - t, len(y_t)))

    X_train = X_t[train_ind, :]
    y_train = y_t[train_ind, :]
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # Fit on training data.
    start = time.time()
    regr = linear_model.LinearRegression(n_jobs = -1) # use all CPUs
    regr.fit(X_train_scaled, y_train)
    end = time.time()
    print 't-{0} took {1} to fit'.format(t, end - start)

    # Transform X_test correctly
    X_test = scaler.transform(X_t[test_ind, :])
    y_test = y_t[test_ind, :]

    # Calculate RMSE.
    y_pred = regr.predict(X_test)
    RMSE = mean_squared_error(y_test, y_pred) ** 0.5
    print RMSE
    RMSE_vec[t] = RMSE
    df = pd.DataFrame({'t': t_vec, 'RMSE': RMSE_vec})
    df.to_csv('sklearn_RMSE.csv', index=False)