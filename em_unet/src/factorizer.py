class Stream:
    def __init__(self, kernel, channels_in, channels_out, subspace, subspace_scale, axis=1):
        self.kernels = []
        self.padding = []
        self.strides = []
        self.filters = []

        n_k = len(kernel)
        assert (n_k == 1 or (n_k == 3 and (kernel[0] == kernel[1] == kernel[2])) ), 'Kernel must be symmetric'

        k = max(kernel)

        t = 0 if subspace <= 1 else subspace

        if axis == 1:
            spatial_kernel = [1, k, k]
            temporal_kernel = [k+t, 1, 1]
        elif axis == 2:
            spatial_kernel = [k, 1, k]
            temporal_kernel = [1, k+t, 1]
        elif axis == 3:
            spatial_kernel = [k, k, 1]
            temporal_kernel = [1, 1, k+t]


        # spatial convolutions (subspace)
        c = channels_in
        for s in range(subspace):
            c = Stream.get_subspace_filters( channels_in, channels_out, [k, k, k], scale=subspace_scale)
            c = min(c, 512)
            #print 's:', s, 'c:', c, 'k:', k
            self.kernels.append( spatial_kernel )
            self.filters.append( c )
            self.strides.append( [1, 1, 1] )
            if axis == 1:
                self.padding.append( [1, 0, 0] )
            elif axis == 2:
                self.padding.append( [0, 1, 0] )
            elif axis == 3:
                self.padding.append( [0, 0, 1] )

        # temporal convolution
        self.kernels.append( temporal_kernel )
        self.filters.append( int(channels_out*subspace_scale) )
        self.strides.append( [1, 1, 1] )

        if axis == 1:
            self.padding.append( [0, 1, 1] )
        elif axis == 2:
            self.padding.append( [1, 0, 1] )
        elif axis == 3:
            self.padding.append( [1, 1, 0] )

        print 'axis:', axis, 'padding:', self.padding


    @staticmethod
    def get_subspace_filters(features_in, features_out, kernel_size, scale=1.0):
        n0  = features_in
        ni  = features_out
        k_t = kernel_size[0]
        k_s = kernel_size[1]
        m = (ni * n0 * k_t * k_s * k_s)/((n0 * 1 * k_s * k_s) + (ni * k_t * 1 * 1))
        #print 'm:', m, 's:', scale
        m = max(m, k_t)
        #m = min(m, int(features_out))
        m = int(m*scale)
        #m = max(m, 3)
        m = min(m, int(features_out))
        #print 'features_out:', features_out, 'features_in:', features_in, 'm:', m, 'kernel:', kernel_size
        return m

class Factorizer:
    def __init__(self, kernel, channels_in, channels_out, subspace, subspace_scale, stream_axes=[1]):
        self.streams = []

        n_streams = len(stream_axes)

        scale = 1.0/float(n_streams)
        scale = scale * subspace_scale
        c_out = channels_out #max( 3, channels_out/n_streams)

        #print 'scale:', scale, c_out
        for stream_axis in stream_axes:
            self.streams.append( Stream(kernel, channels_in, c_out, subspace, subspace_scale=scale, axis=stream_axis ) )
            print 'axes:', stream_axis, channels_in, c_out,  self.streams[-1].kernels, self.streams[-1].filters, self.streams[-1].padding

        #print self.streams


def test_factorizer():
    f = Factorizer([3], 552, 512, 2, 1.0, stream_axes=[1,2])
    for stream in f.streams:
        for kernel, pad, stride, filters, padding in zip(stream.kernels, stream.padding, stream.strides, stream.filters, stream.padding):
            print kernel, pad, stride, filters, padding

#test_factorizer()

