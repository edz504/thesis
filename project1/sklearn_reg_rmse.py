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
all_ind = np.array(xrange(0, N))

# Repeat the train / test split and RMSE evaluation 5 times.
REP = 5
for i in xrange(0, REP):
    train_ind = np.array(xrange(0, T))
    train_ind = np.concatenate(
            (train_ind,
             np.random.choice(all_ind[7:], int(train_proportion * N) - T, replace=False)))
    test_ind = np.array(list(set(all_ind) - set(train_ind)))

    with open('train_inds{0}.txt'.format(i + 1), 'wb') as f:
        for ind in train_ind:
            f.write('{0}\n'.format(int(ind)))

    t_vec = xrange(0, 8)
    RMSE_vec = [None] * len(t_vec)
    for t in t_vec:
        X_cnn_this_t = hkl.load(
            join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
        print 'Loaded X for t-{0}'.format(t)
        y_true = y[t:, ]

        # Split into test and train, making sure to adjust the indices for
        # out of bounds.  Note that each iteration of t removes one index.
        # See confirm_split_train_test.py for example.
        if t > 0:
            train_ind = [ind - 1 for ind in train_ind]
            train_ind.remove(-1)
            train_ind = np.array(train_ind)
            test_ind = [ind - 1 for ind in test_ind]

        X_train = X_cnn_this_t[train_ind, :]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        y_train = y_true[train_ind, :]

        # Fit on training data.
        start = time.time()
        regr = linear_model.LinearRegression(n_jobs = -1) # use all CPUs
        regr.fit(X_train_scaled, y_train)
        end = time.time()
        print 't-{0} took {1} to fit'.format(t, end - start)

        # Transform X_test correctly
        X_test = scaler.transform(X_cnn_this_t[test_ind, :])
        y_test = y_true[test_ind, :]

        # Calculate RMSE.
        y_pred = regr.predict(X_test)
        RMSE = mean_squared_error(y_test, y_pred) ** 0.5
        print RMSE
        RMSE_vec[t] = RMSE
        df = pd.DataFrame({'t': t_vec, 'RMSE': RMSE_vec})
        df.to_csv('sklearn_RMSE{0}.csv'.format(i + 1), index=False)

    print '=======\nFinished repetition {0}\n======='.format(i)