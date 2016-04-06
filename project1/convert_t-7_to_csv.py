from os.path import join
import hickle as hkl
import numpy as np

DATA_DIR = '/home/smile/edzhou/Thesis/data'
y = hkl.load(join(DATA_DIR, 'hkl_train/train_y_gzip.hkl'))
print 'Loaded y'
y_t = y[7:, ]
np.savetxt(join(DATA_DIR,
                'y_t7.csv'), y_t, delimiter=',')
print 'Saved y to csv'

X_t = hkl.load(join(DATA_DIR,
                    'train_cnn_500_t7_gzip.hkl'))
print 'Loaded X'
np.savetxt(join(DATA_DIR,
                'train_cnn_500_t7.csv'), X_t, delimiter=',')
print 'Saved X to csv'