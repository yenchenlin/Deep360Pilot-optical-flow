# This code is currently deprecated, use get_motion_inclusion.py instead

import cv2
import numpy as np
import glob
import argparse
from joblib import Parallel, delayed

W = 1920
H = 960

n_filter = 5
motion_thresh = 0.2
motion_inclusion = 0.3
area_thresh = 0.05

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--domain', dest='domain')
    parser.add_argument('-b', '--boxes', dest='n_boxes')
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


def motion_saliency(flow_mag, n):
    prior = flow_mag / np.max(flow_mag)
    filt = np.ones((n, n))/n/n

    likeli = cv2.filter2D(flow_mag.astype(np.float32), -1, filt)
    likeli = (likeli - likeli.min()) / (likeli.max() - likeli.min())

    return likeli * prior

def get_divide_area_boxes(name):
    FLOW_DIR = 'data/of_' + args.domain + '/' + name + '/'
    BOXES_DIR = 'data/feature_' + args.domain + '_' + \
        str(args.n_boxes) + 'boxes/' + name + '/'

    n_frames = len(glob.glob(FLOW_DIR + '*.png'))

    # init boxes
    clip_boxes_index = 1
    clip_boxes = np.load(BOXES_DIR + 'roislist{:04d}.npy'.format(clip_boxes_index))
    pruned_boxes = np.zeros(clip_boxes.shape)

    for i in xrange(1, n_frames+1):
        print "Flow {}, ".format(i)
        # boxes
        new_clip_boxes_index = (i-1) / 50 + 1
        if clip_boxes_index != new_clip_boxes_index:
            # 1. save pruned_boxes and init a new one
            np.save(BOXES_DIR + 'divide_area_pruned_boxes{:04d}.npy'.format(clip_boxes_index), pruned_boxes)
            pruned_boxes = np.zeros(clip_boxes.shape)

            # 2. update clip_boxes
            clip_boxes_index = new_clip_boxes_index
            clip_boxes = np.load(BOXES_DIR + 'roislist{:04d}.npy'.format(clip_boxes_index))

        # flow
        flow_img = np.array(cv2.imread(FLOW_DIR + '{:06d}.png'.format(i)), dtype=np.float32)
        flow_mag = flow_img[:, :, :2] - 128
        flow_mag = flow_mag / 128

        # compute saliency
        motion_map = motion_saliency(flow_mag, n_filter)
        mot_thresh = min([motion_thresh, 0.7*np.max(motion_map)])
        motion_map = motion_map >= mot_thresh

        frame_boxes = clip_boxes[(i-1) % 50].astype(int)
        for box_id, (xmin, ymin, xmax, ymax) in enumerate(frame_boxes):
            xmin, ymin, xmax, ymax = preprocess_box(xmin, ymin, xmax, ymax)
            box_motion_map = np.asarray(motion_map[ymin:ymax, xmin:xmax, :], dtype=np.float32)

            # box area
            area = float((xmax - xmin) * (ymax - ymin))

            box_motion_magnitude = np.sqrt(np.sum(np.square(box_motion_map), axis=2))
            inclusion = float(np.sum(box_motion_magnitude)) / area

            #box_avg_motion_mag = float(np.sum(box_motion_mag)) / area

            if inclusion >= motion_inclusion:
                print "Box id {}, {} {} {} {}, area: {}, motion: {}".format(
                    box_id, xmin, ymin, xmax, ymax, area, inclusion)
                pruned_boxes[(i-1) % 50][box_id] = np.array([xmin, ymin, xmax, ymax])

    # save latest pruned_boxes
    np.save(BOXES_DIR + 'divide_area_pruned_boxes{:04d}.npy'.format(clip_boxes_index), pruned_boxes)

if __name__ == '__main__':
    args = parse_args()

    NAMEs = sorted(np.load('metadata/metadata_' + args.domain + '.npy').item().keys())
    print NAMEs
    Parallel(n_jobs=5)(delayed(get_divide_area_boxes)(name) for name in NAMEs)
