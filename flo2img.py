import numpy as np
import os
import cv2

"""
This code should not directly be used, function here is called in other file
"""

# A wrapper of converting flow to image
def convert_wrapper(path, outpath, Debug=False):
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.flo'):
            filename = filename.replace('.flo','')

            flow = read_flow(path, filename)
            flow_img = convert_flow(flow, 2.0)

            # NOTE: Change from BGR (OpenCV format) to RGB (Matlab format) to fit Matlab output
            flow_img = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)

            #print "Saving {}.png with shape: {}".format(filename, flow_img.shape)
            cv2.imwrite(outpath + filename + '.png', flow_img)

            if Debug:
                ret = imchecker(outpath + filename)



# Sanity check and comparison if we have matlab version image
def imchecker(filename):

    im_py = cv2.imread(filename + '.png')
    im_mat = cv2.imread(filename + '_mat.png')

    # We'll have presicion problem, but it doesn't affect out results
    result = np.average(im_py - im_mat)
    print "Check result: {}, smaller than 0.01: {}".format(result, result < 0.01)

    return result


# Converting flow to image
def convert_flow(flow, max_flow=8.0):
    assert flow is not None, "Flow should not be None."
    scale = 128.0/max_flow
    mag_flow = np.linalg.norm(flow, axis=2)

    flow = norm_flow(flow, scale)
    mag_flow = norm_flow(mag_flow, scale)

    flow_img = np.concatenate((flow, np.expand_dims(mag_flow,2)),2)
    flow_img = np.round(flow_img)

    return flow_img.astype(np.uint8)


# Flow normalization
def norm_flow(flow, scale):
    flow = flow*scale
    flow = flow + 128
    flow[flow<0.0] = 0
    flow[flow>255.0] = 255

    return flow


# Read in flow file
def read_flow(path, filename):
    flowdata = None
    with open(path + filename + '.flo') as f:
        # Valid .flo file checker
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
        else:
            # Reshape data into 3D array (columns, rows, bands)
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            #print 'Reading {}.flo with shape: ({}, {}, 2)'.format(filename, h, w)
            flowdata = np.fromfile(f, np.float32, count=2*w*h)

            # NOTE: numpy shape(h, w, ch) is opposite to image shape(w, h, ch)
            flowdata = np.reshape(flowdata, (h, w, 2))

    return flowdata


if __name__ == '__main__':

    Debug = True
    path = 'flow/'
    outpath = 'output/'
    convert_wrapper(path, outpath, Debug)
