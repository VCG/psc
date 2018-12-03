from __future__ import print_function
import sys, os, math
import h5py
import numpy as np
from numpy import float32, int32, uint8, dtype
import argparse
import time
# Load PyGreentea
# Relative path to where PyGreentea resides
pygt_path = '../../PyGreentea'
sys.path.append(pygt_path)
import PyGreentea as pygt

import caffe
from caffe import layers as L
from caffe import params as P
from caffe import to_proto
from data import Data
from factorizer import Factorizer


# Set train options
class TrainOptions:
    loss_function = "euclid"
    loss_output_file = "loss.log"
    test_output_file = "test.log"
    test_interval = 2000
    scale_error = 3 #True
    training_method = "affinity"
    recompute_affinity = True
    train_device = 0
    test_device = 1
    test_net= None #'net.prototxt'
    max_iter = int(1.0e4)
    snapshot = int(2000)
    loss_snapshot = int(2000)
    snapshot_prefix = 'net'

class Unet3D:

    @staticmethod
    def start(args):
        #data_shape = [args.depth, args.width, args.height]
        input_shape = [132,132,132]
        output_shape = [44,44,44]

        # Start a network
        net = caffe.NetSpec()

        # Data input layer
        #net.data = L.MemoryData(dim=[1, 1], ntop=1)
        net.data, net.datai = L.MemoryData(dim=[1, 1] + input_shape, ntop=2)

        # Label input layer
        net.label, net.labeli = L.MemoryData(dim=[1, 3] + output_shape, ntop=2, include=[dict(phase=0)])

        # Components label layer 
        net.components, net.componentsi = L.MemoryData(dim=[1, 1] + output_shape, ntop=2, include=[dict(phase=0, stage='malis')])

        # Scale input layer
        net.scale, net.scalei = L.MemoryData(dim=[1, 3] + output_shape, ntop=2, include=[dict(phase=0, stage='euclid')])

        # Silence the not needed data and label integer values
        net.nhood, net.nhoodi = L.MemoryData(dim=[1, 1, 3, 3], ntop=2, include=[dict(phase=0, stage='malis')])

        # Silence the not needed data and label integer values
        net.silence1 = L.Silence(net.datai, net.labeli, net.scalei, ntop=0, include=[dict(phase=0, stage='euclid')])
        net.silence2 = L.Silence(net.datai, net.labeli, net.componentsi, net.nhoodi, ntop=0, include=[dict(phase=0, stage='malis')])
        net.silence3 = L.Silence(net.datai, ntop=0, include=[dict(phase=1)])

        return net

    @staticmethod
    def end(net, x):
        channels_out=3
        x = L.Convolution(x, kernel_size=[1], stride=[1], dilation=[1],num_output=channels_out, pad=[0],
                         param=[dict(lr_mult=1.0),dict(lr_mult=2.0)],weight_filler=dict(type='msra'),bias_filler=dict(type='constant'))

        # Choose output activation functions
        net.prob = L.Sigmoid(x, ntop=1, in_place=False)

        # Choose a loss function and input data, label and scale inputs. Only include it during the training phase (phase = 0)
        net.euclid_loss = L.EuclideanLoss(net.prob, net.label, net.scale, ntop=0, loss_weight=1.0, include=[dict(phase=0, stage='euclid')])
        net.malis_loss = L.MalisLoss(net.prob, net.label, net.components, net.nhood, ntop=0, loss_weight=1.0, include=[dict(phase=0, stage='malis')])

        return net

    @staticmethod
    def vgg_block(x, channels_out):
        relu_slope = 0.005
        for w in range(2):
            x = L.Convolution(x, kernel_size=[3,3,3], stride=[1], dilation=[1],num_output=channels_out, pad=[0],group=1,
                             param=[dict(lr_mult=1.0),dict(lr_mult=2.0)],weight_filler=dict(type='msra'),bias_filler=dict(type='constant'))
            x = L.ReLU(x, in_place=True, negative_slope=relu_slope)
        return x

    @staticmethod
    def baseline(args):

        # Start a network
        net = Unet3D.start(args)

        #UNET config
        net_depth = 3
        net_width = 2
        base_filters = 24
        channels_in = 1
        channels_out = base_filters
          
        # ReLU negative slope
        relu_slope = 0.005

        strategy=[2,2,2]
     
        x = net.data 

        skip = []

        # contracting part
        for d in range(net_depth):
            print ('upsampling...', d)
            channels_out = base_filters*(3**d)
            x = Unet3D.vgg_block(x, channels_out)
            skip.append(x)
            # maxpool
            x = L.Pooling(x, pool=P.Pooling.MAX, kernel_size=strategy, stride=strategy, pad=[0], dilation=[1])
            channels_in = channels_out


        # bridge
        channels_in = channels_out
        channels_out = base_filters*(3**net_depth)
        x = Unet3D.vgg_block(x, channels_out)
    
        # expanding
        channels_in = channels_out
        for d in reversed(range(net_depth)):
            print ('downsampling...', d)
            x = L.Deconvolution(x, 
                                convolution_param=dict(num_output=channels_out, kernel_size=strategy, stride=strategy, pad=[0], 
                                group=channels_out,dilation=[1], bias_term=False,
                                weight_filler=dict(type='constant', value=1)))

            channels_out = base_filters*(3**(d))
            x = L.Convolution(x, kernel_size=[1], stride=[1], dilation=[1],num_output=channels_out, pad=[0],
                             param=[dict(lr_mult=1.0),dict(lr_mult=2.0)],weight_filler=dict(type='msra'),bias_filler=dict(type='constant'))
            x = L.MergeCrop(x, skip[d], forward=[1,1], backward=[1,1], operation=0)

            x = Unet3D.vgg_block(x, channels_out)
            channels_in = channels_out

        net = Unet3D.end( net, x )

        protonet = net.to_proto()
        protonet.name = 'net';

        # Store the network as prototxt
        with open(protonet.name + '.prototxt', 'w') as f:
            print(protonet, file=f)

    @staticmethod
    def vgg_block_approximation(args, model, channels_in, channels_out):

        subspace_scale = 1
        relu_slope = 0.005
        stream_axes = [ i+1 for i in range(args.nstreams)]

        print('stream_axes:', stream_axes)

        n_loops = 1 if args.nsubspace >1 else 2

        for l in range(n_loops):
            stream_convs = []
            factorizer = Factorizer([3], channels_in, channels_out, args.nsubspace, subspace_scale, stream_axes)
            for i, stream in enumerate(factorizer.streams):
                x = model[i]
                for j in range( len(stream.kernels)):
                    kernel = stream.kernels[j]
                    channels_out = stream.filters[j]
                    padding=stream.padding[j]
                    print( kernel, padding)
                    x = L.Convolution(x, kernel_size=kernel,
                            stride=[1], dilation=[1],num_output=channels_out,pad=[0],
                            param=[dict(lr_mult=1.0),dict(lr_mult=2.0)],
                            weight_filler=dict(type='msra'),bias_filler=dict(type='constant'))

                    x = L.ReLU(x, in_place=True, negative_slope=relu_slope)
                stream_convs.append( x )
                x = stream_convs[-1]

            if len(stream_convs) > 1:
                x =  L.Concat( *stream_convs, concat_param=dict(axis=1))

            model = [x for _ in range(args.nstreams)]

        return model[-1]

    @staticmethod
    def approximation(args):
        print ('creating approximation...')

        # Start a network
        net = Unet3D.start(args)

        #UNET config
        net_depth = 3
        net_width = 2
        base_filters = 24
        channels_in = 1
        channels_out = base_filters

        # ReLU negative slope
        relu_slope = 0.005

        strategy=[2,2,2]

        xs = [net.data]
        if args.nstreams > 1:
            xs = L.Split( net.data, ntop=args.nstreams )

        print('xs:', len(xs))
        skip = []

        # contracting part
        for d in range(net_depth):
            print ('upsampling...', d)
            channels_out = base_filters*(3**d)
            x = Unet3D.vgg_block_approximation(args, xs, channels_in, channels_out)
            skip.append(x)
            # maxpool
            x = L.Pooling(x, pool=P.Pooling.MAX, kernel_size=strategy, stride=strategy, pad=[0], dilation=[1])
            channels_in = channels_out
            xs = [x for _ in range(args.nstreams)]

        # bridge
        channels_out = base_filters*(3**net_depth)
        x = Unet3D.vgg_block_approximation(args, xs, channels_in, channels_out)

        # expanding
        channels_in = channels_out
        for d in reversed(range(net_depth)):
            print ('downsampling...', d)
            x = L.Deconvolution(x,
                                convolution_param=dict(num_output=channels_out, kernel_size=strategy, stride=strategy, pad=[0],
                                group=1,dilation=[1], bias_term=False,
                                weight_filler=dict(type='msra'),bias_filler=dict(type='constant')))

            channels_out = base_filters*(3**(d))
            x = L.Convolution(x, kernel_size=[1], stride=[1], dilation=[1],num_output=channels_out, pad=[0],
                             param=[dict(lr_mult=1.0),dict(lr_mult=2.0)],weight_filler=dict(type='msra'),bias_filler=dict(type='constant'))
            x = L.MergeCrop(x, skip[d], forward=[1,1], backward=[1,1], operation=0)

            xs = [x for _ in range(args.nstreams)]
            x = Unet3D.vgg_block_approximation(args, xs, channels_in, channels_out)


        net = Unet3D.end( net, x )
        protonet = net.to_proto()
        protonet.name = 'net';

        # Store the network as prototxt
        with open(protonet.name + '.prototxt', 'w') as f:
            print(protonet, file=f)


    @staticmethod
    def test(args):
        print ('testing...')
        test_dataset, _ = Data.get(args.data_path, args.seg_path, args.data_name, args.seg_name, augment=(args.augment==1))

        modelfile = '%s/net_iter_%d.caffemodel'%(args.output, args.iteration)
        modelproto = '%s/net.prototxt'%(args.output)
        print('modelfile:', modelfile)
        print('modelproto:', modelproto)

        # Set devices
        test_device = 0
        print('Setting devices...')
        pygt.caffe.set_mode_gpu()
        pygt.caffe.set_device(args.test_device)

        # Load model
        print('Loading model...')
        net = pygt.caffe.Net(modelproto, modelfile, pygt.caffe.TEST)

        start_time = time.time()
        # Process
        print('Processing ' + str(len(test_dataset)) + ' volume(s)...')
        preds = pygt.process(net,test_dataset)
        end_time = time.time()

        print("elapsed seconds {0} for z ,y ,x ={1}, {2}, {3}".format(end_time-start_time, preds[0].shape[1],preds[0].shape[2],preds[0].shape[3]))

        for i in range(0,len(test_dataset),1):
            print('Saving ' + test_dataset[i]['name'])
            h5file = '%s/%s_%d-pred.h5'%(args.output, test_dataset[i]['name'], i)
            outhdf5 = h5py.File(h5file, 'w')
            outdset = outhdf5.create_dataset('main', preds[i].shape, np.float32, data=preds[i])
            outhdf5.close()

            h5file = '%s/%s_%d-mask.h5'%(args.output, test_dataset[i]['name'], i)
            outhdf5 = h5py.File(h5file, 'w')
            outdset = outhdf5.create_dataset('main', preds[i+1].shape, np.uint8 , data=preds[i+1])
            outhdf5.close()


    @staticmethod
    def train(args):
        print ('training...')
        train_dataset, test_dataset = Data.get(args.data_path, args.seg_path, args.data_name, args.seg_name, augment=(args.augment==1), transform=(args.transform==1))

        # Set solver options
        print('Initializing solver...')
        solver_config = pygt.caffe.SolverParameter()
        solver_config.train_net = 'net.prototxt'

        solver_config.type = 'Adam'
        solver_config.base_lr = 1e-4
        solver_config.momentum = 0.99
        solver_config.momentum2 = 0.999
        solver_config.delta = 1e-8
        solver_config.weight_decay = 0.000005
        solver_config.lr_policy = 'inv'
        solver_config.gamma = 0.0001
        solver_config.power = 0.75

        solver_config.max_iter = 100000 #nt(2.0e5)
        solver_config.snapshot = int(2000)
        solver_config.snapshot_prefix = 'net'
        solver_config.display = 0

        # Set devices
        print('Setting devices...')
        print(tuple(set((args.train_device, args.test_device))))
        pygt.caffe.enumerate_devices(False)
        #pygt.caffe.set_devices(tuple(set((args.train_device, args.test_device))))
        pygt.caffe.set_devices((int(args.train_device),))

        #pygt.caffe.set_mode_gpu()
        #pygt.caffe.set_device(args.train_device)

        print('devices set...')
        options = TrainOptions()
        options.train_device = args.train_device
        options.test_device = args.test_device

        # First training method
        solver_config.train_state.add_stage('euclid')
        solverstates = pygt.getSolverStates(solver_config.snapshot_prefix)

        print('solver_config.max_iter:', solver_config.max_iter)
        if (len(solverstates) == 0 or solverstates[-1][0] < solver_config.max_iter):
            options.loss_function = 'euclid'
            solver, test_net = pygt.init_solver(solver_config, options)
            if (len(solverstates) > 0):
                print('restoring...', solverstates[-1][1])
                solver.restore(solverstates[-1][1])

            print('euclidean training...')
            pygt.train(solver, test_net, train_dataset, test_dataset, options)


        print('Second training method')
        # Second training method
        solver_config.train_state.set_stage(0, 'malis')
        solverstates = pygt.getSolverStates(solver_config.snapshot_prefix)
        print('solver_config.max_iter:', solver_config.max_iter)
        print('solverstates[-1][0]:', solverstates[-1][0])
        if (solverstates[-1][0] >= solver_config.max_iter):
            # Modify some solver options
            solver_config.max_iter = 300000
            options.loss_function = 'malis'
            # Initialize and restore solver
            solver, test_net = pygt.init_solver(solver_config, options)
            if (len(solverstates) > 0):
                solver.restore(solverstates[-1][1])

            print('malis training...')
            pygt.train(solver, test_net, train_dataset, test_dataset, options)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3D U-Net')
    parser.add_argument('--action', type=str, default='create', help='create | train | test | watershed', required=True)
    parser.add_argument('--data_path', type=str, default='/n/coxfs01/fgonda/experiments/p2d/500-distal/input/original/im_uint8.h5')
    parser.add_argument('--data_name', type=str, default='main')
    parser.add_argument('--seg_path', type=str, default='/n/coxfs01/fgonda/experiments/p2d/500-distal/input/original/groundtruth_seg_thick.h5')
    parser.add_argument('--seg_name', type=str, default='main')
    parser.add_argument('--random_blacks', type=int, default=0)
    parser.add_argument('--train_device', type=int, default=0)
    parser.add_argument('--test_device', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=2000)
    parser.add_argument('--augment', type=int, default=1)
    parser.add_argument('--transform', type=int, default=0)
    parser.add_argument('--output', type=str, default=".")
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--nsubspace', type=int, default='0', help='number of spatial convolutions')
    parser.add_argument('--nstreams', type=int, default='1', help='1 | 2 | 3')
    args = parser.parse_args()


    if args.action == 'create':
        print('creating network...')
        if args.nsubspace == 0 or args.nstreams == 0:
            Unet3D.baseline( args )
        else:
            Unet3D.approximation( args )
    elif args.action == 'train':
        Unet3D.train( args )
    elif args.action == 'test':
        Unet3D.test( args )
    elif args.action == 'watershed':
        Unet3D.waterz( args )

        
