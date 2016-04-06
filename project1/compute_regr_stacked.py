import numpy as np
import pandas as pd
import cPickle
import hickle as hkl
from sklearn import linear_model
from sklearn import preprocessing
from os.path import join

DATA_DIR = '/home/smile/edzhou/Thesis/data'
y = hkl.load(join(DATA_DIR, 'hkl_train/train_y_gzip.hkl'))

t_vec = xrange(0, 8)
AIC_vec = [None] * len(t_vec)
BIC_vec = [None] * len(t_vec)

for t in t_vec:
    X_cnn_this_t = hkl.load(
        join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
    X_scaled = preprocessing.scale(X_cnn_this_t)
    # Start with "t"-th, not 0th data point due to recurrence.
    y_true_mat = y[t:, ]

    logL_sum = 0
    N_sum = 0
    k_sum = 0
    with open(join(DATA_DIR,'sklearn_linear_t{0}.pkl'.format(t)), 'rb') as f:
        regr = cPickle.load(f)
    y_pred_mat = regr.predict(X_scaled)

    for col in xrange(0, y_true_mat.shape[1]):
        y_pred = y_pred_mat[:, col]
        y_true = y_true_mat[:, col]
        these_coef = regr.coef_[col, :]
        SSR = sum([(y_p - y_t) ** 2
                   for y_p, y_t in zip(y_pred, y_true)])
        print SSR
        N = y_pred.shape[0]
        s2 = SSR / N
        logL = -N * 0.5 * np.log(2 * np.pi * s2) - SSR / (2 * s2)
        print logL
        k = sum([1 for coef in these_coef if abs(coef - 0.0) > 1e-10])
        logL_sum += logL
        N_sum += N
        k_sum += k
        print 'AIC for a{0}, t-{1} = {2}'.format(col, t, 2 * k - 2 * logL)
        print 'BIC for a{0}, t-{1} = {2}'.format(col, t, np.log(N_sum) * k - 2 * logL)

    AIC_vec[t] = -2 * logL_sum + 2 * k_sum
    BIC_vec[t] = -2 * logL_sum + (np.log(N_sum) * k_sum)

    print 'Stacked AIC: {0}'.format(AIC_vec[t])
    print 'Stacked BIC: {0}'.format(BIC_vec[t])

    df = pd.DataFrame({'t': t_vec, 'AIC': AIC_vec, 'BIC': BIC_vec})
    df.to_csv('stacked_aic_bic_scale_correctly.csv', index=False)