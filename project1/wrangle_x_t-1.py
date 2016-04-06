import time
import numpy as np
import hickle as hkl

X_cnn = hkl.load('/home/smile/edzhou/Thesis/data/train_cnn_500_gzip.hkl')

# We want our 484815 x 500 matrix to become 484814 x 1000 (we drop the 
# first data point because there is no t-1 for it).

X_cnn_t1 = np.full((484814, 1000), np.nan)
start = time.time()
for i in xrange(1, len(X_cnn_t1) + 1):
    X_cnn_t1[i - 1] = np.append(X_cnn[i], X_cnn[i - 1])
end = time.time()
print 'Took {0} to wrangle t into t - 1'.format(end - start)

hkl.dump(X_cnn_t1,
         '/home/smile/edzhou/Thesis/data/train_cnn_500_t1_gzip.hkl',
         mode='w',
         compression='gzip')