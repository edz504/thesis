import numpy as np
import hickle as hkl

y_full = hkl.load('/home/smile/edzhou/Thesis/data/hkl_train/train_y_gzip.hkl')

# Scale y_full to [0, 1], as was done during CNN training
y_scaled = np.full(y_full.shape, np.nan)
for i, column in enumerate(y_full.T):
    y_scaled[:, i] = (column - min(column)) / (max(column) - min(column))

hkl.dump(y_scaled,
         '''/home/smile/edzhou/Thesis/data/hkl_train/'''
         '''train_y_scaled_gzip.hkl''',
         mode='w',
         compression='gzip')