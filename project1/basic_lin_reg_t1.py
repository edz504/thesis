import time
import numpy as np
import hickle as hkl
import statsmodels.api as sm
import cPickle as pickle

X_cnn_t1 = hkl.load('/home/smile/edzhou/Thesis/data/train_cnn_500_t1_gzip.hkl')
y_scaled = hkl.load('/home/smile/edzhou/Thesis/data/train_y_scaled_gzip.hkl')

num_models = y_scaled.shape[1]
num_to_name = dict(zip(xrange(0, num_models),
                       ['0' + str(i) if i < 10 else i for
                       i in xrange(0, num_models)]))
for i in xrange(0, num_models):
    start = time.time()
    # Start with 1st, not 0th data point due to recurrence.
    model = sm.OLS(y_scaled[1:, i:(i + 1)], X_cnn_t1)
    results = model.fit()
    end = time.time()
    print 'OLS with statsmodels, column {0} took {0} to fit'.format(i,
        end - start)
    with open('''/home/smile/edzhou/Thesis/data/models/'''
              '''linear_model_aff_ind_t1_pvals_{0}.pkl'''.format(
                num_to_name[i + 1]),
              'wb') as f:
        pickle.dump(results.pvalues, f, -1)

model = sm.OLS(y_scaled[1:, 13:14], X_cnn_t1)
results = model.fit()
with open('/home/smile/edzhou/Thesis/data/models/linear_model_aff_ind_t1_pvals_14.pkl',
          'wb') as f:
    pickle.dump(results.pvalues, f, -1)