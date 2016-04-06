import caffe
import leveldb
import numpy as np
import hickle as hkl
from caffe.proto import caffe_pb2

db = leveldb.LevelDB('/home/smile/edzhou/Thesis/data/TORCS_Training_1F')
datum = caffe_pb2.Datum()

N = 484815        # dimensions obtained from examining data
d1 = 3            # dimensions obtained from examining data
d2 = 210
d3 = 280

N_limited = 10000 # save data in chunks of 10000
j = 0             # index into current chunk
num_chunk = 1

# Initialize the first chunk
training_matrix = np.full(shape=(N_limited, d1, d2, d3), fill_value=np.nan)

for i, (key, value) in enumerate(db.RangeIter()):
    datum.ParseFromString(value)
    training_matrix[j] = caffe.io.datum_to_array(datum)
    j += 1

    # Once we fill up the chunk to its limit...
    if j == N_limited:
        # Hickle it
        hkl.dump(training_matrix,
                 'train{0}_gzip.hkl'.format(num_chunk),
                 mode='w',
                 compression='gzip')
        # and create a new matrix to start filling up.
        num_chunk += 1
        training_matrix = np.full(shape=(N_limited, d1, d2, d3), fill_value=np.nan)
        j = 0

# We should have one final training_matrix that is of size 484815 % 10000 = 4815
# at the end that hasn't been dumped, so we need to hickle that as well (and we can
# drop the last 10000 - 4815 rows that have np.nan).
training_matrix = training_matrix[0:4815]
hkl.dump(training_matrix,
                 'train{0}_gzip.hkl'.format(num_chunk),
                 mode='w',
                 compression='gzip')