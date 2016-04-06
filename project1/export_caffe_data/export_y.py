import caffe
import leveldb
import numpy as np
import hickle as hkl
from caffe.proto import caffe_pb2

db = leveldb.LevelDB('/home/smile/edzhou/Thesis/data/TORCS_Training_1F')
datum = caffe_pb2.Datum()

N = 484815        # dimensions obtained from examining data
num_affordance_indicators = 14

y = np.full((N, num_affordance_indicators), np.nan)

for i, (key, value) in enumerate(db.RangeIter()):
    datum.ParseFromString(value)
    y[i] = datum.float_data

hkl.dump(y, '/home/smile/edzhou/Thesis/data/hkl_train/train_y_gzip.hkl', mode='w', compression='gzip')