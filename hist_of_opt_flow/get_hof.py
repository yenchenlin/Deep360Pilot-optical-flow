import numpy as np
import glob
import cv2
import argparse
from HoF import flow_to_hist
from joblib import Parallel, delayed

W = 1920
H = 960

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--domain', dest='domain')
    parser.add_argument('-b', '--boxes', dest='n_boxes', type=int)
    args = parser.parse_args()

    return args

def preprocess_box(xmin, ymin, xmax, ymax):
    if xmin < 0: xmin = 0
    elif xmin > W: xmin = int(xmin - W)
    if ymin < 0: ymin = 0
    elif ymin > H: ymin = int(ymin - H)
    if xmax > W: xmax = int(W)
    elif xmax <= xmin: xmax = xmin+1
    if ymax > H: ymax = int(H)
    elif ymax <= ymin: ymax = ymin+1
    return (xmin, ymin, xmax, ymax)

def get_hof(name):
    print name

    FLOW_DIR = 'data/of_' + args.domain + '/' + name + '/'
    BOXES_DIR = 'data/feature_' + args.domain + \
        '_' + str(args.n_boxes) + 'boxes/' + name + '/'

    n_frames = len(glob.glob(FLOW_DIR + '*.png'))

    # init boxes
    clip_boxes_index = 1
    clip_boxes = np.load(BOXES_DIR + 'roislist{:04d}.npy'.format(clip_boxes_index))

    # init hof
    hof_shape = (50, args.n_boxes, 12)
    hof = np.zeros(hof_shape)

    for i in xrange(1, n_frames+1):
        print "{}, Flow {}, ".format(name, i)
        # boxes
        new_clip_boxes_index = (i-1) / 50 + 1
        if clip_boxes_index != new_clip_boxes_index:
            # 1.1 save hof and init a new one
            np.save(BOXES_DIR + 'hof{:04d}.npy'.format(clip_boxes_index), hof)
            hof = np.zeros(hof_shape)

            # 2.1 update clip_boxes
            clip_boxes_index = new_clip_boxes_index
            clip_boxes = np.load(BOXES_DIR + 'roislist{:04d}.npy'.format(clip_boxes_index))

        flow_img = np.array(cv2.imread(FLOW_DIR + '{:06d}.png'.format(i)), dtype=np.float32)

        frame_boxes = clip_boxes[(i-1) % 50].astype(int)
        for box_id, (xmin, ymin, xmax, ymax) in enumerate(frame_boxes):
            xmin, ymin, xmax, ymax = preprocess_box(xmin, ymin, xmax, ymax)
            box_flow_img = flow_img[ymin:ymax, xmin:xmax, :]
            hof[(i-1) % 50][box_id], _ = flow_to_hist(box_flow_img)

    # save latest hof
    np.save(BOXES_DIR + 'hof{:04d}.npy'.format(clip_boxes_index), hof)

if __name__ == '__main__':
    args = parse_args()
    NAMEs = sorted(np.load('metadata/metadata_' + args.domain + '.npy').item().keys())
    print NAMEs
    Parallel(n_jobs=5)(delayed(get_hof)(name) for name in NAMEs)
