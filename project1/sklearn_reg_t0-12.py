from os.path import join
import time
import hickle as hkl
from sklearn import linear_model
from sklearn import preprocessing
import cPickle

DATA_DIR = '/home/smile/edzhou/Thesis/data'
y = hkl.load(join(DATA_DIR, 'hkl_train/train_y_gzip.hkl'))

for t in xrange(0, 13):
    X_cnn_this_t = hkl.load(
        join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
    print 'Loaded X for t-{0}'.format(t)
    X_scaled = preprocessing.scale(X_cnn_this_t)
    print 'Scaled X for t-{0}'.format(t)

    # sklearn is able to do multivariate, fitting all 14 at once.
    # the coef matrix is then 14 by (500 * (t + 1)), for t = 0, ..., 12.
    regr = linear_model.LinearRegression(n_jobs = -1) # use all CPUs
    
    # Start with "t"-th, not 0th data point due to recurrence.
    y_true = y[t:, ]
    start = time.time()
    regr.fit(X_scaled, y_true)
    end = time.time()
    print 't-{0} took {1} to fit'.format(t, end - start)

    with open(join(DATA_DIR, 'sklearn_linear_t{0}.pkl'.format(t)), 'wb') as f:
        cPickle.dump(regr, f)
    print 'Dumped t-{0}'.format(t)

# Change LinearRegression to be 1 response at a time, possibly will fix segfault?
# for t in xrange(8, 13):
#     X_cnn_this_t = hkl.load(
#         join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
#     print 'Loaded X for t-{0}'.format(t)
    
#     for a in xrange(0, y_scaled.shape[1]): # 14 affordance indicators
#         # Start with "t"-th, not 0th data point due to recurrence.
#         y_true = y_scaled[t:, a:(a + 1)].ravel()
#         start = time.time()
#         regr = linear_model.LinearRegression(copy_X=False, n_jobs = -1) # use all CPUs
#         regr.fit(X_cnn_this_t, y_true)
#         end = time.time()
#         print 'linear t-{0}, a{1} took {2} to fit'.format(t, a, end - start)

#         with open(join(DATA_DIR, 'sklearn_linear_t{0}_a{1}.pkl'.format(t, a)),
#                   'wb') as f:
#             cPickle.dump(sgd, f)
#         print 'Dumped t-{0}, a{1}'.format(t, a)

# for t in xrange(0, 13):
#     X_cnn_this_t = hkl.load(
#         join(DATA_DIR, 'train_cnn_500_t{0}_gzip.hkl'.format(t)))
#     print 'Loaded X for t-{0}'.format(t)

#     # SGD cannot do multivariate, so we need to loop through affordance.
#     sgd = linear_model.SGDRegressor(loss='squared_loss', penalty='elasticnet')

#     for a in xrange(0, y_scaled.shape[1]): # 14 affordance indicators
#         # Start with "t"-th, not 0th data point due to recurrence.
#         y_true = y_scaled[t:, a:(a + 1)].ravel()
#         start = time.time()
#         sgd.fit(X_cnn_this_t, y_true)
#         end = time.time()
#         print 't-{0}, a{1} took {2} to fit'.format(t, a, end - start)

#         with open(join(DATA_DIR, 'sklearn_sgd_t{0}_a{1}.pkl'.format(t, a)),
#                   'wb') as f:
#             cPickle.dump(sgd, f)
#         print 'Dumped t-{0}, a{1}'.format(t, a)
