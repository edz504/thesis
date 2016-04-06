import numpy as np
import pandas as pd
import cPickle
import hickle as hkl
from sklearn import linear_model
from os.path import join

DATA_DIR = '/home/smile/edzhou/Thesis/data'
y_scaled = hkl.load(join(DATA_DIR, 'train_y_scaled_gzip.hkl'))

AIC_vec = [None] * 12
BIC_vec = [None] * 12
t_vec = xrange(0, 13)

for t in t_vec:
    X_cnn_this_t = hkl.load(
        join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))

    logL_sum = 0
    N_sum = 0
    k_sum = 0

    for a in xrange(0, y_scaled.shape[1]):
        # Start with "t"-th, not 0th data point due to recurrence.
        y_true = y_scaled[t:, a:(a + 1)].ravel()

        with open(join(DATA_DIR,
                       'sklearn_sgd_t{0}_a{1}.pkl'.format(t, a)),
                  'rb') as f:
            sgd = cPickle.load(f)
        y_pred = sgd.predict(X_cnn_this_t)

        SSR = sum([(y_p - y_t) ** 2
                   for y_p, y_t in zip(y_pred, y_true)])
        N = y_pred.shape[0]
        s2 = SSR / N
        logL = -N * 0.5 * np.log(2 * np.pi * s2) - SSR / (2 * s2)
        k = X_cnn_this_t.shape[1]

        logL_sum += logL
        N_sum += N
        k_sum += k
        print 'Finished prediction and computation for a = {0}, t-{1}'.format(a, t)

    AIC_vec[t] = logL_sum - k_sum
    BIC_vec[t] = logL_sum - (np.log(N_sum) / 2) * k_sum

    print 'Stacked AIC: {0}'.format(AIC_vec[t])
    print 'Stacked BIC: {0}'.format(BIC_vec[t])

df = pd.DataFrame({'t': t_vec, 'AIC': AIC_vec, 'BIC': BIC_vec})
df.to_csv('stacked_aic_bic.csv', index=False)