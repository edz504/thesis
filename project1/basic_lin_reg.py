import time
import numpy as np
import hickle as hkl
import statsmodels.api as sm
import cPickle as pickle
# from sklearn import linear_model
# from linear_regression_coef import LinearRegression

X_cnn = hkl.load('/home/smile/edzhou/Thesis/data/train_cnn_500_gzip.hkl')
y_scaled = hkl.load('/home/smile/edzhou/Thesis/data/train_y_scaled_gzip.hkl')

num_models = y_scaled.shape[1]
num_to_name = dict(zip(xrange(0, num_models),
                       ['0' + str(i) if i < 10 else i for
                       i in xrange(0, num_models)]))

for i in xrange(0, y_scaled.shape[1]):
    start = time.time()
    model = sm.OLS(y_scaled[:, i:(i + 1)], X_cnn)
    results = model.fit()
    end = time.time()
    print 'OLS with statsmodels, column {0} took {0} to fit'.format(i, end - start)
    with open('''/home/smile/edzhou/Thesis/data/models/'''
              '''linear_model_aff_ind_{0}.pkl'''.format(
              num_to_name[i + 1]),
              'wb') as f:
        pickle.dump(results, f, -1)

# sklearn
# start = time.time()
# regr = LinearRegression()
# regr.fit(X_cnn, y_scaled[:, 0])
# end = time.time()
# print 'Basic OLS with sklearn took {0} to fit'.format(end - start)