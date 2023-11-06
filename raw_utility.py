# Bruce A. Maxwell
# June 2023
# Library of useful functions for dealing with raw images


import sys
import rawpy
import numpy as np
import math
import cv2
import os

# scale factor
SCALE_FACTOR = 1
AUTO_BRIGHT_OFF = False
AUTO_BRIGHT_THR = 0.001
SAMPLES = 8000

# read and postprocess the raw image
# todo: add an optional argument for size of small side
def processRAW( path, scale_factor=SCALE_FACTOR, auto_bright_off = AUTO_BRIGHT_OFF, auto_bright_thr = AUTO_BRIGHT_THR ):

    # read the file
    with rawpy.imread(path) as raw_file:
        num_bits = int(math.log(raw_file.white_level + 1, 2))
        rgb = raw_file.postprocess( gamma=(1,1), no_auto_bright=AUTO_BRIGHT_OFF, auto_bright_thr = AUTO_BRIGHT_THR, output_bps=16, use_camera_wb=True)

        # reduce the size
        dim = ( rgb.shape[1] // SCALE_FACTOR, rgb.shape[0] // SCALE_FACTOR )
        resized_rgb = cv2.resize( rgb, dim, interpolation = cv2.INTER_AREA )

        log_rgb = resized_rgb.astype("float32")
        log_rgb[log_rgb != 0] = np.log( log_rgb[log_rgb != 0] )

        
    return resized_rgb, log_rgb


def processRAW_img(img, scale_factor=SCALE_FACTOR, auto_bright_off=AUTO_BRIGHT_OFF, auto_bright_thr=AUTO_BRIGHT_THR):
    # read the file
    with img:
        num_bits = int(math.log(raw_file.white_level + 1, 2))
        rgb = raw_file.postprocess(gamma=(1, 1), no_auto_bright=AUTO_BRIGHT_OFF, auto_bright_thr=AUTO_BRIGHT_THR,
                                   output_bps=16, use_camera_wb=True)

        # reduce the size
        dim = (rgb.shape[1] // SCALE_FACTOR, rgb.shape[0] // SCALE_FACTOR)
        resized_rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)

        log_rgb = resized_rgb.astype("float32")
        log_rgb[log_rgb != 0] = np.log(log_rgb[log_rgb != 0])

    return resized_rgb, log_rgb

# read and process a tiff image
def processTIF( path, scale_factor=SCALE_FACTOR ):

    # read the file
    rgb = cv2.imread( path, cv2.IMREAD_UNCHANGED )
    rgb = cv2.cvtColor( rgb, cv2.COLOR_BGR2RGB )

    # reduce the size
    dim = ( rgb.shape[1] // scale_factor, rgb.shape[0] // scale_factor )
    resized_rgb = cv2.resize( rgb, dim, interpolation = cv2.INTER_AREA )

    log_rgb = resized_rgb.astype("float32")
    log_rgb[log_rgb != 0] = np.log( log_rgb[log_rgb != 0] )

    return resized_rgb, log_rgb


# read and process the jpg image
def processJPG( path, scale_factor=SCALE_FACTOR ):

    # read the file
    rgb = cv2.imread( path )
    rgb = cv2.cvtColor( rgb, cv2.COLOR_BGR2RGB )

    # reduce the size
    dim = ( rgb.shape[1] // scale_factor, rgb.shape[0] // scale_factor )
    resized_rgb = cv2.resize( rgb, dim, interpolation = cv2.INTER_AREA )

    log_rgb = resized_rgb.astype("float32")
    log_rgb[log_rgb != 0] = np.log( log_rgb[log_rgb != 0] )

    return resized_rgb, log_rgb
    
def main(argv):

    if len(argv) < 2:
        print("usage: python %s <image path>" % (argv[0]) )
        return

    # grab the file path
    filename = argv[1]

    suffix = filename.split(".")[1].lower()
    logdiv = 11.4
    if  suffix == 'jpg':
        rgb, logrgb = processJPG( filename, scale_factor = 4 )
        disp_image = np.copy(rgb)
        logdiv = 5.7
    elif suffix == 'tif':
        rgb, logrgb = processTIF( filename, scale_factor = 4 )
        disp_image = rgb / 256
        disp_image = disp_image.astype("uint8")
    else:
        rgb, logrgb = processRAW( filename, scale_factor = 4 )
        disp_image = rgb / 256
        disp_image = disp_image.astype("uint8")

    # convert to 8-bit
    print("rgb shape:", rgb.shape)
    disp_image = cv2.cvtColor( disp_image, cv2.COLOR_BGR2RGB )

    displog_image = (logrgb / logdiv) * 255
    displog_image = displog_image.astype( "uint8" )
    displog_image = cv2.cvtColor( displog_image, cv2.COLOR_BGR2RGB )
    
    cv2.imshow( "RGB", disp_image )

    cv2.imshow( "log RGB", displog_image )
    cv2.imwrite( "logimage.png", displog_image)

    cv2.waitKey()

    return


if __name__ == "__main__":
    main(sys.argv)


        
        
