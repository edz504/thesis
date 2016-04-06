import hickle as hkl
from wrangle_x_general import wrangle_x

X_cnn = hkl.load(
    '/home/smile/edzhou/Thesis/data/train_cnn_500_gzip.hkl')
print 'Finished loading X_cnn'
DATA_DIR = DATA_DIR = '/home/smile/edzhou/Thesis/data/'

# wrangle_x(X_cnn, 2, DATA_DIR)
# wrangle_x(X_cnn, 3, DATA_DIR)
# wrangle_x(X_cnn, 4, DATA_DIR)
# wrangle_x(X_cnn, 5, DATA_DIR)
# wrangle_x(X_cnn, 6, DATA_DIR)

# wrangle_x(X_cnn, 7, DATA_DIR)
# wrangle_x(X_cnn, 8, DATA_DIR)
# wrangle_x(X_cnn, 9, DATA_DIR)

wrangle_x(X_cnn, 10, DATA_DIR)
print 'Finished 10'
wrangle_x(X_cnn, 11, DATA_DIR)
print 'Finished 11'
wrangle_x(X_cnn, 12, DATA_DIR)
print 'Finished 12'
