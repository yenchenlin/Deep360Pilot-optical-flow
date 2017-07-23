import argparse
import numpy as np
import glob
from joblib import Parallel, delayed

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--domain', dest='domain')
    parser.add_argument('-b', '--boxes', dest='n_boxes', type=int)
    args = parser.parse_args()

    return args

args = parse_args()
NAMEs = sorted(np.load('metadata/metadata_' + args.domain + '.npy').item().keys())
BOX_FEATURE = 'divide_area_pruned_boxes'

def gen_pruned_features(name):
    print name
    feature_dir = 'data/feature_' + args.domain + \
        '_' + str(args.n_boxes) + 'boxes/' + name + '/'
    n_clips = len(glob.glob(feature_dir + BOX_FEATURE + '*.npy'))
    for clip in xrange(1, n_clips+1):
        pruned_boxes = np.load(feature_dir + BOX_FEATURE + '{:04d}.npy'.format(clip)) # (50, args.n_boxes, 4)
        roisavg = np.load(feature_dir + 'roisavg{:04d}.npy'.format(clip)) # (50, args.n_boxes, 512)

        pruned_roisavg = np.zeros((50, args.n_boxes, 512))
        for frame in xrange(50):
            for box_id in xrange(args.n_boxes):
                if not np.array_equal(pruned_boxes[frame][box_id], np.zeros((4))):
                    pruned_roisavg[frame][box_id] = roisavg[frame][box_id]

        np.save('{}pruned_roisavg{:04d}'.format(feature_dir, clip), pruned_roisavg)

Parallel(n_jobs=5)(delayed(gen_pruned_features)(name) for name in NAMEs)
