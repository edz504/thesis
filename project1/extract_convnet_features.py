import os
import time
import numpy as np
import hickle as hkl
from lasagne import layers
from nolearn.lasagne import NeuralNet
import cPickle as pickle
import theano

# Load the model
with open('/home/smile/edzhou/Thesis/data/models/model1.pkl', 'rb') as f:
    convnet = pickle.load(f)

# Define the function that obtains output passed from input to hidden layer 
# (right before output)
dense_layer = layers.get_output(convnet.layers_['hidden5'], deterministic=True)
input_var = convnet.layers_['input'].input_var
f_dense = theano.function([input_var], dense_layer)

# For each training chunk (size = 10000 x 3 x 210 x 280), pass through this
# function and obtain the feature representation.  Our feature representation
# should, at the end, be 484815 x 500.
X_cnn = np.full((484815, 500), np.nan)
DATA_LOCATION = '/home/smile/edzhou/Thesis/data/hkl_train/'
X_filenames = sorted([fn for fn in os.listdir(DATA_LOCATION) if fn != 'train_y_gzip.hkl'])
start = 0
MINI_CHUNK_SIZE = 1000
for i, X_filename in enumerate(X_filenames):
    time1 = time.time()
    this_X = hkl.load(os.path.join(DATA_LOCATION, X_filename))
    print 'Loaded chunk {0} in {1}'.format(i, time.time() - time1)
    this_X_32 = this_X.astype('float32')
    end = start + len(this_X_32)
    time2 = time.time()
    # We can't pass a 10000 x 3 x 210 x 280 matrix to f_dense or else we run
    # out of memory, so we need to operate one (mini)-chunk at a time.
    this_X_cnn = np.full((len(this_X_32), 500), np.nan)
    mini_start = 0
    while mini_start < len(this_X_32):
        mini_end = min(mini_start + MINI_CHUNK_SIZE, len(this_X_32))
        this_X_cnn[mini_start:mini_end, ] = f_dense(
            this_X_32[mini_start:mini_end, ])
        mini_start = mini_end
    X_cnn[start:end, ] = this_X_cnn
    print 'Extracted feature reps for chunk {0} in {1}'.format(
        i, time.time() - time2)
    start = end

hkl.dump(X_cnn, '/home/smile/edzhou/Thesis/data/train_cnn_500_gzip.hkl',
                 mode='w',
                 compression='gzip')