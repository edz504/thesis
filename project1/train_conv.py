import os
import numpy as np
import hickle as hkl
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import time
import cPickle as pickle

convnet = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 210, 280),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=14, output_nonlinearity=None,

    update_learning_rate=0.00001,
    update_momentum=0.9,

    regression=True,
    max_epochs=100, # change this to 1000 if possible
    verbose=1,
    )

DATA_LOCATION = '/home/smile/edzhou/Thesis/data/hkl_train/'
y_full = hkl.load(os.path.join(DATA_LOCATION, 'train_y_gzip.hkl'))
X_filenames = sorted([fn for fn in os.listdir(DATA_LOCATION) if fn != 'train_y_gzip.hkl'])

# Scale y_full to [0, 1]
y_scaled = np.full(y_full.shape, np.nan)
for i, column in enumerate(y_full.T):
    y_scaled[:, i] = (column - min(column)) / (max(column) - min(column))

start = 0
for i, X_filename in enumerate(X_filenames):
    time1 = time.time()
    this_X = hkl.load(os.path.join(DATA_LOCATION, X_filename))
    print 'X chunk {0} loaded, took {1}'.format(i, time.time() - time1)
    end = start + len(this_X)
    this_y = y_scaled[start:end]

    time2 = time.time()
    this_X_32 = this_X.astype('float32')
    this_y_32 = this_y.astype('float32')
    print 'Chunk {0} took {1} to cast to float32'.format(i, time.time() - time2)

    time3 = time.time()
    convnet.fit(this_X_32, this_y_32)
    print 'Took {0} to train one chunk, 100 epochs'.format(time.time() - time3)

    start = end

with open('/home/smile/edzhou/Thesis/data/models/model1.pkl', 'wb') as f:
    pickle.dump(convnet, f, -1)