import time
import numpy as np
import hickle as hkl

# Load data (t - 7 from BIC selection).
t = 7
X_cnn_this_t = hkl.load(
'/home/smile/edzhou/Thesis/data/train_cnn_500_t{0}_gzip.hkl'.format(
  t))
print 'Loaded X for t-{0}'.format(t)
y_scaled = hkl.load(
  '/home/smile/edzhou/Thesis/data/train_y_scaled_gzip.hkl')
y_scaled_t = y_scaled[t:, ]

# Split into 80/20 train/test, and hickle train and test for both X
# and y_scaled.  Also hickle the indices to be rigorous.



