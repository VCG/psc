import numpy as np
import waterz

aff = np.zeros((3,10,10,10))
#seg_gt = np.ones((3,10,10,10))

# affinities is a [3,depth,height,width] numpy array of float32
affinities = np.zeros((3,10,10,10))

# evaluation: vi/rand
seg_gt = None

aff_thresholds = [0.005, 0.995]
seg_thresholds = [0.1, 0.3, 0.6]
seg = waterz.waterz(aff, seg_thresholds, merge_function='aff50_his256',aff_threshold=aff_thresholds, gt=seg_gt)
