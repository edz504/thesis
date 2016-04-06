import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2

db = leveldb.LevelDB('/home/smile/edzhou/Thesis/data/TORCS_Training_1F')
datum = caffe_pb2.Datum()


with open('count_results.txt', 'wb') as f:
    for i, (key, value) in enumerate(db.RangeIter()):
        if i % 50000 == 0:
            f.write('progress = {0}\n'.format(i))
    f.write('========\nNumber of data: {0}\n'.format(i + 1))
