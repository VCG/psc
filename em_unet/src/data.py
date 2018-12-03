import sys, os, math
import scipy.misc
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
from os.path import join
import pdb

# Load PyGreentea
# Relative path to where PyGreentea resides
pygt_path = '../../PyGreentea'
sys.path.append(pygt_path)
import PyGreentea as pygt


class Data:

    @staticmethod
    def get(data_path, seg_path, data_name='main', seg_name='stack', augment=False, transform=False):
        print('loading dataset...', data_path)

        filename = data_path.split('/')[-1]
        filename = filename.split('.')[0]

        test_dataset = []
        train_dataset = []

        p_data = data_path
        p_seg = seg_path
        train_dataset.append({})
        train_dataset[-1]['name'] = filename
        train_dataset[-1]['nhood'] = pygt.malis.mknhood3d()
        train_dataset[-1]['data'] = np.array( h5py.File( p_data )[ data_name ], dtype=np.float32)/(2.**8)
        train_dataset[-1]['components'] = np.array( h5py.File( p_seg )[ seg_name ] )
        train_dataset[-1]['label'] = pygt.malis.seg_to_affgraph(train_dataset[-1]['components'],train_dataset[-1]['nhood'])

        if transform:
            train_dataset[-1]['transform'] = {}
            train_dataset[-1]['transform']['scale'] = (0.8,1.2)
            train_dataset[-1]['transform']['shift'] = (-0.2,0.2)

        if augment:
            print 'augmenting...'
            train_dataset = pygt.augment_data_simple(train_dataset,trn_method='affinity')

        for iset in range(len(train_dataset)):
            train_dataset[iset]['data'] = train_dataset[iset]['data'][None,:] # add a dummy dimension
            train_dataset[iset]['components'] = train_dataset[iset]['components'][None,:]
            print(train_dataset[iset]['name'] + str(iset) + ' shape:' + str(train_dataset[iset]['data'].shape))

        return train_dataset,test_dataset


def extract( a, prefix, offset=0 ):
    print a.shape
    for i in range(10):
        if len(a.shape) > 3:
            img = a[0,i+offset,:,:]
        else:
            img = a[i+offset,:,:]
        scipy.misc.imsave('./extract/%s_%d.tif'%(prefix,i), img)

def test_data():
    train, test = Data.cremi(augment=False)
    extract( train[-1]['data'], 'data')
    extract( train[-1]['label'], 'label')


#test_data()
