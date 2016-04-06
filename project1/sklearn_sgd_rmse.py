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

N = len(y)
train_proportion = 0.8

# Should have made that down to 12
# t_vec = xrange(0, 8)
# RMSE_vec = [None] * len(t_vec)

# for t in t_vec:
#     X_t = hkl.load(
#         join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
#     y_t = y[t:, ]

#     train_ind = np.array(xrange(0, int(train_proportion * N) - t))
#     test_ind = np.array(xrange(int(train_proportion * N) - t, len(y_t)))

#     X_train = X_t[train_ind, :]
#     y_train_mat = y_t[train_ind, :]
#     scaler = preprocessing.StandardScaler().fit(X_train)
#     X_train_scaled = scaler.transform(X_train)
#     X_test = scaler.transform(X_t[test_ind, :])
#     y_test_mat = y_t[test_ind, :]
#     y_pred_mat = np.zeros(y_test_mat.shape)

#     start = time.time()
#     for a in xrange(0, y.shape[1]):
#         y_train = y_train_mat[:, a:(a + 1)].ravel()
#         sgd = linear_model.SGDRegressor(alpha=100, n_iter=20)
#         sgd.fit(X_train_scaled, y_train)
#         y_pred_mat[:, a] = sgd.predict(X_test)
#         print a
#         sys.stdout.flush()
#     RMSE = mean_squared_error(y_test_mat, y_pred_mat) ** 0.5
#     print RMSE
#     RMSE_vec[t] = RMSE
#     df = pd.DataFrame({'t': t_vec, 'RMSE': RMSE_vec})
#     df.to_csv('sklearn_sgd_RMSE.csv', index=False)
#     end = time.time()
#     print 't-{0} took {1} to 80/20 fit + RMSE'.format(t, end - start)


# t_vec = xrange(8, 13)
# RMSE_vec = [None] * len(t_vec)

# for t in t_vec:
#     X_t = hkl.load(
#         join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
#     y_t = y[t:, ]

#     train_ind = np.array(xrange(0, int(train_proportion * N) - t))
#     test_ind = np.array(xrange(int(train_proportion * N) - t, len(y_t)))

#     X_train = X_t[train_ind, :]
#     y_train_mat = y_t[train_ind, :]
#     scaler = preprocessing.StandardScaler().fit(X_train)
#     X_train_scaled = scaler.transform(X_train)
#     X_test = scaler.transform(X_t[test_ind, :])
#     y_test_mat = y_t[test_ind, :]
#     y_pred_mat = np.zeros(y_test_mat.shape)

#     start = time.time()
#     for a in xrange(0, y.shape[1]):
#         y_train = y_train_mat[:, a:(a + 1)].ravel()
#         sgd = linear_model.SGDRegressor(alpha=100, n_iter=20)
#         sgd.fit(X_train_scaled, y_train)
#         y_pred_mat[:, a] = sgd.predict(X_test)
#         print a
#     RMSE = mean_squared_error(y_test_mat, y_pred_mat) ** 0.5
#     print RMSE
#     RMSE_vec[t - 8] = RMSE
#     df = pd.DataFrame({'t': t_vec, 'RMSE': RMSE_vec})
#     df.to_csv('sklearn_sgd_RMSE_8-12.csv', index=False)
#     end = time.time()
#     print 't-{0} took {1} to 80/20 fit + RMSE'.format(t, end - start)

# t_vec = xrange(10, 13)
# RMSE_vec = [None] * len(t_vec)

# for t in t_vec:
#     X_t = hkl.load(
#         join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
#     y_t = y[t:, ]

#     train_ind = np.array(xrange(0, int(train_proportion * N) - t))
#     test_ind = np.array(xrange(int(train_proportion * N) - t, len(y_t)))

#     X_train = X_t[train_ind, :]
#     y_train_mat = y_t[train_ind, :]
#     scaler = preprocessing.StandardScaler(copy=False)
#     scaler.fit(X_train)
#     scaler.transform(X_train, copy=False)
#     X_test = X_t[test_ind, :]
#     scaler.transform(X_test)
#     y_test_mat = y_t[test_ind, :]
#     y_pred_mat = np.zeros(y_test_mat.shape)

#     start = time.time()
#     for a in xrange(0, y.shape[1]):
#         y_train = y_train_mat[:, a:(a + 1)].ravel()
#         sgd = linear_model.SGDRegressor(alpha=100, n_iter=20)
#         sgd.fit(X_train, y_train)
#         y_pred_mat[:, a] = sgd.predict(X_test)
#         print a
#     RMSE = mean_squared_error(y_test_mat, y_pred_mat) ** 0.5
#     print RMSE
#     RMSE_vec[t - 8] = RMSE
#     df = pd.DataFrame({'t': t_vec, 'RMSE': RMSE_vec})
#     df.to_csv('sklearn_sgd_RMSE_10-12.csv', index=False)
#     end = time.time()
#     print 't-{0} took {1} to 80/20 fit + RMSE'.format(t, end - start)

t = 12
X_t = hkl.load(
    join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
y_t = y[t:, ]

train_ind = np.array(xrange(0, int(train_proportion * N) - t))
test_ind = np.array(xrange(int(train_proportion * N) - t, len(y_t)))

X_train = X_t[train_ind, :]
y_train_mat = y_t[train_ind, :]
scaler = preprocessing.StandardScaler(copy=False)
scaler.fit(X_train)
scaler.transform(X_train, copy=False)
X_test = X_t[test_ind, :]
scaler.transform(X_test)
y_test_mat = y_t[test_ind, :]
y_pred_mat = np.zeros(y_test_mat.shape)

start = time.time()
for a in xrange(0, y.shape[1]):
    y_train = y_train_mat[:, a:(a + 1)].ravel()
    sgd = linear_model.SGDRegressor(alpha=100, n_iter=20)
    sgd.fit(X_train, y_train)
    y_pred_mat[:, a] = sgd.predict(X_test)
    print a
RMSE = mean_squared_error(y_test_mat, y_pred_mat) ** 0.5
print RMSE
df = pd.DataFrame({'t': [12], 'RMSE': [RMSE]})
df.to_csv('sklearn_sgd_RMSE_12.csv', index=False)
end = time.time()
print 't-{0} took {1} to 80/20 fit + RMSE'.format(t, end - start)