import os, sys, inspect, gc
import h5py
import numpy as np
from scipy import io
import math
import threading
import png
from Crypto.Random.random import randint
import numpy.random
import pdb

# Determine where PyGreentea is
pygtpath = os.path.normpath(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])))

# Determine where PyGreentea gets called from
cmdpath = os.getcwd()

sys.path.append(pygtpath)
sys.path.append(cmdpath)


from numpy import float32, int32, uint8

# Load the configuration file
import config

# Load the setup module
import setup

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Direct call to PyGreentea, set up everything
if __name__ == "__main__":
    if (pygtpath != cmdpath):
        os.chdir(pygtpath)
    
    if (os.geteuid() != 0):
        print(bcolors.WARNING + "PyGreentea setup should probably be executed with root privileges!" + bcolors.ENDC)
    
    if config.install_packages:
        print(bcolors.HEADER + ("==== PYGT: Installing OS packages ====").ljust(80,"=") + bcolors.ENDC)
        setup.install_dependencies()
    
    print(bcolors.HEADER + ("==== PYGT: Updating Caffe/Greentea repository ====").ljust(80,"=") + bcolors.ENDC)
    setup.clone_caffe(config.caffe_path, config.clone_caffe, config.update_caffe)
    
    print(bcolors.HEADER + ("==== PYGT: Updating Malis repository ====").ljust(80,"=") + bcolors.ENDC)
    setup.clone_malis(config.malis_path, config.clone_malis, config.update_malis)
    
    if config.compile_caffe:
        print(bcolors.HEADER + ("==== PYGT: Compiling Caffe/Greentea ====").ljust(80,"=") + bcolors.ENDC)
        setup.compile_caffe(config.caffe_path)
    
    if config.compile_malis:
        print(bcolors.HEADER + ("==== PYGT: Compiling Malis ====").ljust(80,"=") + bcolors.ENDC)
        setup.compile_malis(config.malis_path)
        
    if (pygtpath != cmdpath):
        os.chdir(cmdpath)
    
    print(bcolors.OKGREEN + ("==== PYGT: Setup finished ====").ljust(80,"=") + bcolors.ENDC)
    sys.exit(0)

#pdb.set_trace()
setup.setup_paths(config.caffe_path, config.malis_path)
setup.set_environment_vars()

# Import Caffe
import caffe as caffe

# Import the network generator
import network_generator as netgen

# Import Malis
import malis as malis


# Wrapper around a networks set_input_arrays to prevent memory leaks of locked up arrays
class NetInputWrapper:
    
    def __init__(self, net, shapes):
        self.net = net
        self.shapes = shapes
        self.dummy_slice = np.ascontiguousarray([0]).astype(float32)
        self.inputs = []
        for i in range(0,len(shapes)):
            # Pre-allocate arrays that will persist with the network
            self.inputs += [np.zeros(tuple(self.shapes[i]), dtype=float32)]
                
    def setInputs(self, data):      
        #pdb.set_trace()
        #print('---setinputs---')
        for i in range(0,len(self.shapes)):
            #print(i, self.shapes)
            #print(data[i].shape) 
            #print(self.inputs[i].shape)
            #print('===')
            np.copyto(self.inputs[i], np.ascontiguousarray(data[i]).astype(float32))
            self.net.set_input_arrays(i, self.inputs[i], self.dummy_slice)
                  

# Transfer network weights from one network to another
def net_weight_transfer(dst_net, src_net):
    # Go through all source layers/weights
    for layer_key in src_net.params:
        # Test existence of the weights in destination network
        if (layer_key in dst_net.params):
            # Copy weights + bias
            for i in range(0, min(len(dst_net.params[layer_key]), len(src_net.params[layer_key]))):
                np.copyto(dst_net.params[layer_key][i].data, src_net.params[layer_key][i].data)
        
class ClassWeight:
    def __init__(self, aug_datalbl, recent_iter):
	self.pred_thd = 0.0
	self.alpha = 2
	nz0idx = np.where(aug_datalbl[0]['label'] == 0)
	self.const_wt0 = aug_datalbl[0]['label'].size*1.0/len(nz0idx[2])
	nz1idx = np.where(aug_datalbl[0]['label'] == 1)
	self.const_wt1 = aug_datalbl[0]['label'].size*1.0/len(nz1idx[2])
	
	self.nclass = np.unique(aug_datalbl[0]['label']).size

	self.class_ind = []
	for i in range(0,(len(aug_datalbl))):
	    self.class_ind.append([])
	    actual_labels = aug_datalbl[i]['label']
	    
	    indmat = []
	    for cc in range(0,self.nclass):
		indmat.append([])
		indmat[-1] = (actual_labels == cc).astype('uint8')
	    
	    self.class_ind[-1] = indmat
	
	
	
	#pdb.set_trace()
	self.class_weights = []
	weight_filename =  'weights_itr'+str(recent_iter)+'.h5'
	if os.path.exists(weight_filename):
	    fp = h5py.File(weight_filename)
	    ndsets = fp.keys()
	    for i in range(len(ndsets)):
		dataset_name = 'stack'+str(i)
		self.class_weights.append([])
		self.class_weights[i] = np.array(fp[dataset_name]).astype(np.float32) 
		
	    fp.close()
	  
	else:
	    for i in range(0,(len(aug_datalbl))):
		self.class_weights.append([])
		self.class_weights[i] = (self.const_wt0 * self.class_ind[i][0]) + (self.const_wt1 * self.class_ind[i][1])
		self.class_weights[i] = self.class_weights[i].astype(np.float32)
		
	## # toufiq debug
	#pdb.set_trace()
	##for i in range(0,(len(aug_datalbl))):
	#savename = 'tst-weights520.h5'
	#fp = h5py.File(savename,'w')
	#fp.create_dataset('stack1',data=self.class_weights[1])
	#fp.create_dataset('stack5',data=self.class_weights[5])
	#fp.create_dataset('stack10',data=self.class_weights[10])
	#fp.create_dataset('stack15',data=self.class_weights[15])
	#fp.close()
	#pdb.set_trace()
	    
	    
    def recompute_weight(self, trn_pred_array, trn_itr):  
	#pdb.set_trace()
	for i in range(0,(len(trn_pred_array))):
	    
	    pred0_diff = (trn_pred_array[i] - self.pred_thd) 
	    wt0 = self.class_weights[i] + (self.alpha * pred0_diff) 
	    wt0_clipped = np.clip(wt0, self.const_wt0, 50*self.const_wt0 ) # membrane weight cannot be less than cyto weights
	    
	    self.class_weights[i] = (wt0_clipped * self.class_ind[i][0] ) + ( self.const_wt1 * self.class_ind[i][1] )
	    
	## # toufiq debug
	#savename = 'weights_itr'+str(trn_itr)+'.h5'
	#fp = h5py.File(savename,'w')
	#for i in range(len(self.class_weights)):
	    #dataset_name = 'stack'+str(i)
	    #fp.create_dataset(dataset_name,data=self.class_weights[i],compression='gzip',compression_opts=9)
	    
	#fp.close()
	
	
	    
	    
        
def normalize(dataset, newmin=-1, newmax=1):
    maxval = dataset
    while len(maxval.shape) > 0:
        maxval = maxval.max(0)
    minval = dataset
    while len(minval.shape) > 0:
        minval = minval.min(0)
    return ((dataset - minval) / (maxval - minval)) * (newmax - newmin) + newmin


def getSolverStates(prefix):
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print files
    solverstates = []
    for file in files:
        if(prefix+'_iter_' in file and '.solverstate' in file):
            solverstates += [(int(file[len(prefix+'_iter_'):-len('.solverstate')]),file)]
    return sorted(solverstates)
            
def getCaffeModels(prefix):
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    print files
    caffemodels = []
    for file in files:
        if(prefix+'_iter_' in file and '.caffemodel' in file):
            caffemodels += [(int(file[len(prefix+'_iter_'):-len('.caffemodel')]),file)]
    return sorted(caffemodels)
            

def error_scale(data, factor_low, factor_high):
    scale = np.add((data >= 0.5) * factor_high, (data < 0.5) * factor_low)
    return scale

def error_scale_overall(data, weight_vec):
    #pdb.set_trace()
    scale = np.zeros(data.shape)
    nclass = weight_vec.shape[0]
    for cc in range(nclass):
	binary_indicator = np.array(data == cc)
	scale += ((1.0/weight_vec[cc]) * binary_indicator)
        
    return scale
    
def class_balance_distribution(label_array):
    #pdb.set_trace()
    nclass = np.unique(label_array).shape[0]
    weight_vec = []
    for cc in range(nclass):
	binary_indicator = np.array(label_array == cc)
        frac_cc = np.clip(binary_indicator.mean(),0.05,0.95) #for binary labels
        weight_vec.append(frac_cc)
        
    return(np.array(weight_vec))


def count_affinity(dataset):
    aff_high = np.sum(dataset >= 0.5)
    aff_low = np.sum(dataset < 0.5)
    return aff_high, aff_low


def border_reflect(dataset, border):
    return np.pad(dataset,((border, border)),'reflect')

def augment_data_simple(dataset,trn_method='affinity'):
    nset = len(dataset)
    for iset in range(nset):
        for reflectz in range(2):
            for reflecty in range(2):
                for reflectx in range(2):
                    for swapxy in range(2):
                        for swapzx in range(2):

                            if reflectz==0 and reflecty==0 and reflectx==0 and swapxy==0:
                                continue

                            dataset.append({})

                            if trn_method == 'affinity':
                                dataset[-1]['name'] = dataset[iset]['name']
                                dataset[-1]['nhood'] = dataset[iset]['nhood']
                                dataset[-1]['data'] = dataset[iset]['data'][:]
                                dataset[-1]['components'] = dataset[iset]['components'][:]

                                if reflectz:
                                    dataset[-1]['data']         = dataset[-1]['data'][::-1,:,:]
                                    dataset[-1]['components']   = dataset[-1]['components'][::-1,:,:]

                                if reflecty:
                                    dataset[-1]['data']         = dataset[-1]['data'][:,::-1,:]
                                    dataset[-1]['components']   = dataset[-1]['components'][:,::-1,:]

                                if reflectx:
                                    dataset[-1]['data']         = dataset[-1]['data'][:,:,::-1]
                                    dataset[-1]['components']   = dataset[-1]['components'][:,:,::-1]

                                if swapxy:
                                    dataset[-1]['data']         = dataset[-1]['data'].transpose((0,2,1))
                                    dataset[-1]['components']   = dataset[-1]['components'].transpose((0,2,1))

                                if swapzx:
                                    dataset[-1]['data']         = dataset[-1]['data'].transpose((2,1,0))
                                    dataset[-1]['components']   = dataset[-1]['components'].transpose((2,1,0))

                                dataset[-1]['label'] = malis.seg_to_affgraph(dataset[-1]['components'],dataset[-1]['nhood'])


                            dataset[-1]['reflectz']=reflectz
                            dataset[-1]['reflecty']=reflecty
                            dataset[-1]['reflectx']=reflectx
                            dataset[-1]['swapxy']=swapxy
    #pdb.set_trace()
    return dataset

def old_augment_data_simple(dataset,trn_method='affinity'):
    nset = len(dataset)
    for iset in range(nset):
        for reflectz in range(2):
            for reflecty in range(2):
                for reflectx in range(2):
                    for swapxy in range(2):

                        if reflectz==0 and reflecty==0 and reflectx==0 and swapxy==0:
                            continue
			
                        dataset.append({})
                        if trn_method == 'affinity':
			    dataset[-1]['name'] = dataset[iset]['name']
			    dataset[-1]['nhood'] = dataset[iset]['nhood']
			    dataset[-1]['data'] = dataset[iset]['data'][:]
			    dataset[-1]['components'] = dataset[iset]['components'][:]

			    if reflectz:
				dataset[-1]['data']         = dataset[-1]['data'][::-1,:,:]
				dataset[-1]['components']   = dataset[-1]['components'][::-1,:,:]

			    if reflecty:
				dataset[-1]['data']         = dataset[-1]['data'][:,::-1,:]
				dataset[-1]['components']   = dataset[-1]['components'][:,::-1,:]

			    if reflectx:
				dataset[-1]['data']         = dataset[-1]['data'][:,:,::-1]
				dataset[-1]['components']   = dataset[-1]['components'][:,:,::-1]

			    if swapxy:
				dataset[-1]['data']         = dataset[-1]['data'].transpose((0,2,1))
				dataset[-1]['components']   = dataset[-1]['components'].transpose((0,2,1))

			    dataset[-1]['label'] = malis.seg_to_affgraph(dataset[-1]['components'],dataset[-1]['nhood'])

                        elif trn_method == 'pixel':
			    dataset[-1]['name'] = dataset[iset]['name']
			    dataset[-1]['nhood'] = dataset[iset]['nhood']
			    dataset[-1]['data'] = dataset[iset]['data'][:]
			    dataset[-1]['label'] = dataset[iset]['label'][:]
			    #dataset[-1]['components'] = dataset[iset]['components'][:]

			    if reflectz:
				dataset[-1]['data']         = dataset[-1]['data'][::-1,:,:]
				if len(dataset[-1]['label'].shape)==3:
				    dataset[-1]['label']   = dataset[-1]['label'][::-1,:,:]
				elif len(dataset[-1]['label'].shape)==4: 
				    dataset[-1]['label']   = dataset[-1]['label'][:,::-1,:,:]

			    if reflecty:
				dataset[-1]['data']         = dataset[-1]['data'][:,::-1,:]
				if len(dataset[-1]['label'].shape)==3:
				    dataset[-1]['label']   = dataset[-1]['label'][:,::-1,:]
				elif len(dataset[-1]['label'].shape)==4: 
				    dataset[-1]['label']   = dataset[-1]['label'][:,:,::-1,:]

			    if reflectx:
				dataset[-1]['data']         = dataset[-1]['data'][:,:,::-1]
				if len(dataset[-1]['label'].shape)==3:
				    dataset[-1]['label']   = dataset[-1]['label'][:,:,::-1]
				elif len(dataset[-1]['label'].shape)==4: 
				    dataset[-1]['label']   = dataset[-1]['label'][:,:,:,::-1]

			    if swapxy:
				dataset[-1]['data']         = dataset[-1]['data'].transpose((0,2,1))
				if len(dataset[-1]['label'].shape)==3:
				    dataset[-1]['label']   = dataset[-1]['label'].transpose((0,2,1))
				elif len(dataset[-1]['label'].shape)==4: 
				    dataset[-1]['label']   = dataset[-1]['label'].transpose((0,1,3,2))

			    #dataset[-1]['label'] = malis.seg_to_affgraph(dataset[-1]['components'],dataset[-1]['nhood'])
			    
			####dataset[-1]['transform'] = dataset[iset]['transform']
			    
			dataset[-1]['reflectz']=reflectz
			dataset[-1]['reflecty']=reflecty
			dataset[-1]['reflectx']=reflectx
			dataset[-1]['swapxy']=swapxy
			
			
    #pdb.set_trace()
    return dataset

    
def augment_data_elastic(dataset,ncopy_per_dset):
    dsetout = []
    nset = len(dataset)
    for iset in range(nset):
        for icopy in range(ncopy_per_dset):
            reflectz = np.random.rand()>.5
            reflecty = np.random.rand()>.5
            reflectx = np.random.rand()>.5
            swapxy = np.random.rand()>.5

            dataset.append({})
            dataset[-1]['reflectz']=reflectz
            dataset[-1]['reflecty']=reflecty
            dataset[-1]['reflectx']=reflectx
            dataset[-1]['swapxy']=swapxy

            dataset[-1]['name'] = dataset[iset]['name']
            dataset[-1]['nhood'] = dataset[iset]['nhood']
            dataset[-1]['data'] = dataset[iset]['data'][:]
            dataset[-1]['components'] = dataset[iset]['components'][:]

            if reflectz:
                dataset[-1]['data']         = dataset[-1]['data'][::-1,:,:]
                dataset[-1]['components']   = dataset[-1]['components'][::-1,:,:]

            if reflecty:
                dataset[-1]['data']         = dataset[-1]['data'][:,::-1,:]
                dataset[-1]['components']   = dataset[-1]['components'][:,::-1,:]

            if reflectx:
                dataset[-1]['data']         = dataset[-1]['data'][:,:,::-1]
                dataset[-1]['components']   = dataset[-1]['components'][:,:,::-1]

            if swapxy:
                dataset[-1]['data']         = dataset[-1]['data'].transpose((0,2,1))
                dataset[-1]['components']   = dataset[-1]['components'].transpose((0,2,1))

            # elastic deformations

            dataset[-1]['label'] = malis.seg_to_affgraph(dataset[-1]['components'],dataset[-1]['nhood'])

    return dataset

    
def slice_data(data, offsets, sizes):
    if (len(offsets) == 1):
        return data[offsets[0]:offsets[0] + sizes[0]]
    if (len(offsets) == 2):
        return data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1]]
    if (len(offsets) == 3):
        return data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1], offsets[2]:offsets[2] + sizes[2]]
    if (len(offsets) == 4):
        return data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1], offsets[2]:offsets[2] + sizes[2], offsets[3]:offsets[3] + sizes[3]]


def set_slice_data(data, insert_data, offsets, sizes):
    if (len(offsets) == 1):
        data[offsets[0]:offsets[0] + sizes[0]] = insert_data
    if (len(offsets) == 2):
        data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1]] = insert_data
    if (len(offsets) == 3):
        data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1], offsets[2]:offsets[2] + sizes[2]] = insert_data
    if (len(offsets) == 4):
        data[offsets[0]:offsets[0] + sizes[0], offsets[1]:offsets[1] + sizes[1], offsets[2]:offsets[2] + sizes[2], offsets[3]:offsets[3] + sizes[3]] = insert_data


def sanity_check_net_blobs(net):
    for key in net.blobs.keys():
        dst = net.blobs[key]
        data = np.ndarray.flatten(dst.data[0].copy())
        print 'Blob: %s; %s' % (key, data.shape)
        failure = False
        first = -1
        for i in range(0,data.shape[0]):
            if abs(data[i]) > 1000:
                failure = True
                if first == -1:
                    first = i
                print 'Failure, location %d; objective %d' % (i, data[i])
        print 'Failure: %s, first at %d, mean %3.5f' % (failure,first,np.mean(data))
        if failure:
            break
        
def dump_feature_maps(net, folder):
    for key in net.blobs.keys():
        dst = net.blobs[key]
        norm = normalize(dst.data[0], 0, 255)
        # print(norm.shape)
        for f in range(0,norm.shape[0]):
            outfile = open(folder+'/'+key+'_'+str(f)+'.png', 'wb')
            writer = png.Writer(norm.shape[2], norm.shape[1], greyscale=True)
            # print(np.uint8(norm[f,:]).shape)
            writer.write(outfile, np.uint8(norm[f,:]))
            outfile.close()
                
        
def get_net_input_specs(net, test_blobs = ['data', 'label', 'scale', 'label_affinity', 'affinty_edges']):
    
    shapes = []
    
    # The order of the inputs is strict in our network types
    for blob in test_blobs:
        if (blob in net.blobs):
            shapes += [[blob, np.shape(net.blobs[blob].data)]]
        
    return shapes

def get_spatial_io_dims(net):
    out_primary = 'label'
    
    if ('prob' in net.blobs):
        out_primary = 'prob'
    
    shapes = get_net_input_specs(net, test_blobs=['data', out_primary])
        
    dims = len(shapes[0][1]) - 2
    print(dims)
    
    input_dims = list(shapes[0][1])[2:2+dims]
    output_dims = list(shapes[1][1])[2:2+dims]
    padding = [input_dims[i]-output_dims[i] for i in range(0,dims)]
    
    
    return input_dims, output_dims, padding

def get_fmap_io_dims(net):
    out_primary = 'label'
    
    if ('prob' in net.blobs):
        out_primary = 'prob'
    
    shapes = get_net_input_specs(net, test_blobs=['data', out_primary])
    
    input_fmaps = list(shapes[0][1])[1]
    output_fmaps = list(shapes[1][1])[1]
    
    return input_fmaps, output_fmaps
    
    
def get_net_output_specs(net):
    return np.shape(net.blobs['prob'].data)


def process(net, data_arrays, shapes=None, net_io=None):    
    input_dims, output_dims, input_padding = get_spatial_io_dims(net)
    fmaps_in, fmaps_out = get_fmap_io_dims(net)
    dims = len(output_dims)
    #pdb.set_trace()
    if (shapes == None):
        shapes = []
        # Raw data slice input         (n = 1, f = 1, spatial dims)
        shapes += [[1,fmaps_in] + input_dims]
        
    if (net_io == None):
        net_io = NetInputWrapper(net, shapes)
            
    dst = net.blobs['prob']
    dummy_slice = [0]
    
    pred_arrays = []
    for i in range(0, len(data_arrays)):
        data_array = data_arrays[i]['data']
        data_dims = len(data_array.shape)
        
        offsets = []        
        in_dims = []
        out_dims = []
        for d in range(0, dims):
            offsets += [0]
            in_dims += [data_array.shape[data_dims-dims+d]]
            out_dims += [data_array.shape[data_dims-dims+d] - input_padding[d]]
        
        plane_id = 0
        if dims==2:
	    in_dims = [data_array.shape[1]] + in_dims
	    out_dims = [data_array.shape[1]] + out_dims
	    offsets = [plane_id] + offsets
	    
        pred_array = np.zeros(tuple([fmaps_out] + out_dims))
        
        #pdb.set_trace()
        while(True):
	    if dims==3:
		data_slice = slice_data(data_array, [0] + offsets, [fmaps_in] + [output_dims[di] + input_padding[di] for di in range(0, dims)])
	    elif dims==2:
		data_slice = slice_data(data_array, [0] + offsets, [fmaps_in,1] + [output_dims[di] + input_padding[di] for di in range(0, dims)])
		
            net_io.setInputs([data_slice])
            net.forward()
            output = dst.data[0].copy()
            
            
            if dims==3:
		set_slice_data(pred_array, output, [0] + offsets, [fmaps_out] + output_dims)
		print offsets
		print output.mean()
	    elif dims==2:
		output = np.expand_dims(output,axis=1)
		set_slice_data(pred_array, output, [0] + offsets, [fmaps_out,1] + output_dims)
            
            incremented = False
	    #pdb.set_trace()
	    #if offsets[0]==124:
		#print offsets
		#print output.mean()
            for d in range(0, dims):
                ##if (offsets[dims - 1 - d] == out_dims[dims - 1 - d] - output_dims[dims - 1 - d]):
                    ### Reset direction
                    ##offsets[dims - 1 - d] = 0
                ##else:
                    ### Increment direction
                    ##offsets[dims - 1 - d] = min(offsets[dims - 1 - d] + output_dims[dims - 1 - d], out_dims[dims - 1 - d] - output_dims[dims - 1 - d])
                    ##incremented = True
                    ##break
                ninp_dims = len(in_dims)    
                if (offsets[ninp_dims - 1 - d] == out_dims[ninp_dims - 1 - d] - output_dims[dims - 1 - d]):
                    # Reset direction
                    offsets[ninp_dims - 1 - d] = 0
                else:
                    # Increment direction
                    offsets[ninp_dims - 1 - d] = min(offsets[ninp_dims - 1 - d] + output_dims[dims - 1 - d], out_dims[ninp_dims - 1 - d] - output_dims[dims - 1 - d])
                    incremented = True
                    break
                    
            # Processed the whole input block, or, in case of 2D, the slice
            if not incremented:
		if dims==2 and plane_id < (in_dims[0]-1):
		    print offsets
		    print output.mean()
		    
		    plane_id = plane_id + 1
		    offsets[0] = plane_id
		    incremented = True
		else:
		    break

	#pdb.set_trace()
	mask = np.zeros(tuple([fmaps_out] + in_dims))    
	if dims==3:
	    startz = (input_dims[0]-output_dims[0])/2;
	    endz = in_dims[0] - startz
	    starty = (input_dims[1]-output_dims[1])/2;
	    endy = in_dims[1] - starty
	    startx = (input_dims[2]-output_dims[2])/2;
	    endx = in_dims[2] - startx
	    
	    mask[:,startz:endz, starty:endy, startx:endx] = 1
	  
	elif dims==2:
	    starty = (input_dims[0]-output_dims[0])/2;
	    endy = in_dims[1] - starty
	    startx = (input_dims[1]-output_dims[1])/2;
	    endx = in_dims[2] - startx
	    mask[:,:, starty:endy, startx:endx] = 1
	
        #pred_arrays += [pred_array]
        pred_arrays += [pred_array]
        pred_arrays += [mask]
            
    return pred_arrays
      
        
    # Wrapper around a networks 
class TestNetEvaluator:
    
    def __init__(self, test_net, train_net, data_arrays, options):
        self.options = options
        self.test_net = test_net
        self.train_net = train_net
        self.data_arrays = data_arrays
        self.thread = None
        
        input_dims, output_dims, input_padding = get_spatial_io_dims(self.test_net)
        fmaps_in, fmaps_out = get_fmap_io_dims(self.test_net)       
        self.shapes = []
        self.shapes += [[1,fmaps_in] + input_dims]
        self.net_io = NetInputWrapper(self.test_net, self.shapes)
            
    def run_test(self, iteration):
        caffe.select_device(self.options.test_device, False)
        self.pred_arrays = process(self.test_net, self.data_arrays, shapes=self.shapes, net_io=self.net_io)

        for i in range(0, 1):
        #for i in range(0, len(self.data_arrays)):
            if ('name' in self.data_arrays[i]):
                h5file = self.data_arrays[i]['name'] + '.h5'
            else:
                h5file = 'test_out_' + repr(i) + '.h5'
            outhdf5 = h5py.File(h5file, 'w')
            outdset = outhdf5.create_dataset('main', self.pred_arrays[i*2].shape, np.float32, data=self.pred_arrays[i*2])
            # outdset.attrs['nhood'] = np.string_('-1,0,0;0,-1,0;0,0,-1')
            outhdf5.close()
            
        count=0 
        #pdb.set_trace()
        self.pred_arrays_samesize = []
        for i in range(0, len(self.pred_arrays),2):
	    pred_array1 = self.pred_arrays[i]
            pred_mask = self.pred_arrays[i+1] 
            
	    nz_idx = np.where(pred_mask[0,...]>0)
	    
	    pred_array1_samesize = np.zeros(pred_mask.shape).astype(np.float32)
	    for cc in range(pred_array1_samesize.shape[0]):
		pred_array1_samesize[cc,nz_idx[0],nz_idx[1],nz_idx[2]] = pred_array1[cc,...].ravel()
	    
	    self.pred_arrays_samesize.append([])
	    self.pred_arrays_samesize[-1] = pred_array1_samesize

	    
    def evaluate(self, iteration):
        # Test/wait if last test is done
        if not(self.thread is None):
            try:
                self.thread.join()
            except:
                self.thread = None
        # Weight transfer
        net_weight_transfer(self.test_net, self.train_net)
        # Run test
	# # Toufiq -- debug check
        self.run_test(iteration)
        
        #self.thread = threading.Thread(target=self.run_test, args=[iteration])
        #self.thread.start()
        
                

def init_solver(solver_config, options):
    caffe.set_mode_gpu()
    caffe.select_device(options.train_device, False)
   
    solver_inst = caffe.get_solver(solver_config)
    
    if (options.test_net == None):
        return (solver_inst, None)
    else:
        return (solver_inst, init_testnet(options.test_net, test_device=options.test_device))
    
def init_testnet(test_net, trained_model=None, test_device=0):
    caffe.set_mode_gpu()
    caffe.select_device(test_device, False)
    if(trained_model == None):
        return caffe.Net(test_net, caffe.TEST)
    else:
        return caffe.Net(test_net, trained_model, caffe.TEST)

    
def train(solver, test_net, data_arrays, train_data_arrays, options, random_blacks=False):
    caffe.select_device(options.train_device, False)
    
    net = solver.net
    
    #pdb.set_trace()
    clwt=None
    test_eval = None
    if options.scale_error == 2:
	clwt = ClassWeight(data_arrays, solver.iter)
        test_eval = TestNetEvaluator(test_net, net, data_arrays, options)
    
    test_eval2 = None
    if (options.test_net != None):
        test_eval2 = TestNetEvaluator(test_net, net, train_data_arrays, options)
        
    input_dims, output_dims, input_padding = get_spatial_io_dims(net)
    fmaps_in, fmaps_out = get_fmap_io_dims(net)

    dims = len(output_dims)
    losses = []
    
    shapes = []
    # Raw data slice input         (n = 1, f = 1, spatial dims)
    shapes += [[1,fmaps_in] + input_dims]
    # Label data slice input    (n = 1, f = #edges, spatial dims)
    shapes += [[1,fmaps_out] + output_dims]
    
    if (options.loss_function == 'malis'):
        # Connected components input   (n = 1, f = 1, spatial dims)
        shapes += [[1,1] + output_dims]
    if (options.loss_function == 'euclid'):
        # Error scale input   (n = 1, f = #edges, spatial dims)
        shapes += [[1,fmaps_out] + output_dims]
    # Nhood specifications         (n = #edges, f = 3)
    if (('nhood' in data_arrays[0]) and (options.loss_function == 'malis')):
        shapes += [[1,1] + list(np.shape(data_arrays[0]['nhood']))]

    net_io = NetInputWrapper(net, shapes)
    
    weight_vec = []
    if (options.loss_function == 'softmax' or options.loss_function == 'euclid') and options.scale_error == 1:
	#pdb.set_trace()
	weight_vec = class_balance_distribution(data_arrays[0]['label'])
	#weight_vec[2] = weight_vec[1]*4.0 #for 3 class, inversed during weighting
    
    #pdb.set_trace()
    # Loop from current iteration to last iteration
    for i in range(solver.iter, solver.max_iter):
        
        if (options.test_net != None and i % options.test_interval == 0 and i>1):
	    #pdb.set_trace()
            test_eval2.evaluate(i)
            if options.scale_error == 2:
		test_eval.evaluate(i)
		clwt.recompute_weight(test_eval.pred_arrays_samesize, i)

        
        # First pick the dataset to train with
        dataset = randint(0, len(data_arrays) - 1)

	if dims==3:
	    offsets = []
	    for j in range(0, dims):
		offsets.append(randint(0, data_arrays[dataset]['data'].shape[1+j] - (output_dims[j] + input_padding[j])))
		
	    # These are the raw data elements
	    #pdb.set_trace()        
	    data_slice = slice_data(data_arrays[dataset]['data'], [0]+offsets, [fmaps_in]+[output_dims[di] + input_padding[di] for di in range(0, dims)])
	    label_slice = slice_data(data_arrays[dataset]['label'], [0] + [offsets[di] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], [fmaps_out] + output_dims)
	    
	    if options.scale_error ==2 and clwt != None:
		weight_slice = slice_data(clwt.class_weights[dataset], [0] + [offsets[di] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], [fmaps_out] + output_dims)

	elif dims==2:
	    offsets = []
	    offsets.append(randint(0,data_arrays[dataset]['data'].shape[1]-1))
	    for j in range(0, dims):
		offsets.append(randint(0, data_arrays[dataset]['data'].shape[2+j] - (output_dims[j] + input_padding[j])))
		
	    # These are the raw data elements
	    #pdb.set_trace()        
	    data_slice = slice_data(data_arrays[dataset]['data'], [0]+offsets, [fmaps_in,1]+[output_dims[di] + input_padding[di] for di in range(0, dims)])
	    label_slice = slice_data(data_arrays[dataset]['label'], [0, offsets[0]] + [offsets[di+1] + int(math.ceil(input_padding[di] / float(2))) for di in range(0, dims)], [fmaps_out,1] + output_dims)
	  
	    data_slice = np.squeeze(data_slice)
	    label_slice = np.squeeze(label_slice)
        #offsets=np.zeros(dims);
	    if (data_slice.shape[0]<1) or (label_slice.shape[0]<2):
		pp=1
		pdb.set_trace()
        

	#pdb.set_trace()
	#if(np.unique(label_slice).shape[0]<2):
	 #   continue;	
	
        # transform the input
        # this code assumes that the original input pixel values are scaled between (0,1)
        if 'transform' in data_arrays[dataset]:
            # print('Pre:',(data_slice.min(),data_slice.mean(),data_slice.max()))
            data_slice_mean = data_slice.mean()
            lo, hi = data_arrays[dataset]['transform']['scale']
            data_slice = data_slice_mean + (data_slice-data_slice_mean)*np.random.uniform(low=lo,high=hi)
            lo, hi = data_arrays[dataset]['transform']['shift']
            data_slice = data_slice + np.random.uniform(low=lo,high=hi)
            # print('Post:',(data_slice.min(),data_slice.mean(),data_slice.max()))
            data_slice = np.clip(data_slice, 0.0, 0.95)
      
	#pdb.set_trace()


        if random_blacks and np.random.random() < 0.25:
            data_slice[ randint( 0, data_slice.shape[0]-1 ),...] = 0.0

	
        if options.loss_function == 'malis':
	    components_slice,ccSizes = malis.connected_components_affgraph(label_slice.astype(int32), data_arrays[dataset]['nhood'])
            # Also recomputing the corresponding labels (connected components)
            net_io.setInputs([data_slice, label_slice, components_slice, data_arrays[0]['nhood']])
            
        if options.loss_function == 'euclid':
            ###if(options.scale_error == True):
                ###frac_pos = np.clip(label_slice.mean(),0.05,0.95) #for binary labels
                ###w_pos = 1.0/(2.0*frac_pos)
                ###w_neg = 1.0/(2.0*(1.0-frac_pos))
            ###else:
                ###w_pos = 1
                ###w_neg = 1          
                
            ###net_io.setInputs([data_slice, label_slice, error_scale(label_slice,w_neg,w_pos)])
	    if(options.scale_error == 3):
		frac_pos = np.clip(label_slice.mean(),0.01,0.99) #for binary labels
		w_pos = 1.0/(2.0*frac_pos)
		w_neg = 1.0/(2.0*(1.0-frac_pos))
		
		net_io.setInputs([data_slice, label_slice, error_scale(label_slice,w_neg,w_pos)])
		
            elif(options.scale_error == 1):
		frac_pos = weight_vec[0]
		w_pos = 1./frac_pos

		label_weights = error_scale_overall(label_slice, weight_vec)
		net_io.setInputs([data_slice, label_slice, label_weights])
		
	    elif options.scale_error == 2:
		net_io.setInputs([data_slice, label_slice, weight_slice])
	      
            elif options.scale_error == 0:
		net_io.setInputs([data_slice, label_slice])

        if options.loss_function == 'softmax':
	    net_io.setInputs([data_slice, label_slice])
        
        # Single step
        loss = solver.step(1)
        # sanity_check_net_blobs(net)
        
        while gc.collect():
            pass


        if (options.loss_function == 'euclid' or options.loss_function == 'euclid_aniso') and options.scale_error ==1 :
            print("[Iter %i] Loss: %f, frac_pos=%f, w_pos=%f" % (i,loss,frac_pos,w_pos))
        else:
            print("[Iter %i] Loss: %f" % (i,loss))
        # TODO: Store losses to file
        losses += [loss]

        if hasattr(options, 'loss_snapshot') and ((i % options.loss_snapshot) == 0):
            io.savemat('loss.mat',{'loss':losses})

