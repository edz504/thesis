import os
import time
import numpy as np
import hickle as hkl

def wrangle_x(X_cnn, num_prev, DATA_DIR):
    nrows = X_cnn.shape[0] - num_prev
    ncols = X_cnn.shape[1] * (num_prev + 1)
    X_cnn_prev = np.full((nrows, ncols), np.nan)
    start = time.time()
    for i in xrange(num_prev, nrows + num_prev):
        total = X_cnn[i]
        for j in xrange(1, num_prev + 1):
            total = np.append(total, X_cnn[i - j])
        X_cnn_prev[i - num_prev] = total
    end = time.time()
    print 'Took {0} to wrangle t into t - {1}'.format(
        end - start, num_prev)

    start = time.time()
    hkl.dump(X_cnn_prev,
             os.path.join(
                DATA_DIR,
                'train_cnn_500_t{0}_gzip.hkl'.format(num_prev)),
             mode='w',
             compression='gzip')
    end = time.time()
    print 'Took {0} to hickle t - {1}'.format(end - start, num_prev)