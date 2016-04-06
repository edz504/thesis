import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2

# db = leveldb.LevelDB('/home/smile/edzhou/Thesis/data/TORCS_baseline_testset/TORCS_Caltech_1F_Testing')
# datum = caffe_pb2.Datum()

# with open('caltech_test_count.txt', 'wb') as f:
#    for i, (key, value) in enumerate(db.RangeIter()):
#        if i % 50000 == 0:
#            f.write('progress = {0}\n'.format(i))
#    f.write('========\nNumber of data: {0}\n'.format(i + 1))

db = leveldb.LevelDB('/home/smile/edzhou/Thesis/data/TORCS_baseline_testset/TORCS_GIST_1F_Testing')
with open('gist_test_count.txt', 'wb') as f:
    for i, (key, value) in enumerate(db.RangeIter()):
        if i % 50000 == 0:
            f.write('progress = {0}\n'.format(i))
    f.write('========\nNumber of data: {0}\n'.format(i + 1))
