from os.path import join
import hickle as hkl
import numpy as np

DATA_DIR = '/home/smile/edzhou/Thesis/data'
t = 7
X_t = hkl.load(join(DATA_DIR,
                    'train_cnn_500_t{0}_gzip.hkl'.format(t)))

X_expand = np.zeros((len(X_t),
                     X_t.shape[1] * 3))

for i in xrange(0, X_t.shape[1]):
    X_expand[:, 3 * i] = X_t[:, i]
    X_expand[:, 3 * i + 1] = X_t[:, i] ** 2
    X_expand[:, 3 * i + 2] = X_t[:, i] ** 3
    if i % 40 == 0:
        print i

hkl.dump(X_expand,
         join(DATA_DIR, 'expand_poly3_t-7_gzip.hkl'),
         mode='w',
         compression='gzip')
