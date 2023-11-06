import imageio
import numpy as np
import cv2

# imageio.plugins.freeimage.download()

def float_to_tiff(pseudo_flt_img):
    max_16_bit_val = 65535
    sixteen_bit_image = (np.floor_divide((pseudo_flt_img * max_16_bit_val), 1)).astype(np.uint16)
    return sixteen_bit_image

def save_tiff(sixteen_bit_image, fpath):
    imageio.imsave(fpath, sixteen_bit_image)

def tiff_to_log(sixteen_bit_image):
    sixteen_bit_image = sixteen_bit_image.astype(np.float32)
    sixteen_bit_image[sixteen_bit_image!=0] = np.log(sixteen_bit_image[sixteen_bit_image!=0])
    return sixteen_bit_image

def save_log(log_image, fpath):
    #fpath = "./output6"
    imageio.imsave(fpath, log_image)

def srgb_to_xyz(srgb_img, max_val=1):
  # Max val is an optional parameter to scale the image to [0, 1].
  # Images must be scaled to [0, 1].
  srgb_img = srgb_img / max_val
  low_mask = srgb_img <= 0.04045
  high_mask = srgb_img > 0.04045
  srgb_img[low_mask] /= 12.92
  srgb_img[high_mask] = (((srgb_img[high_mask]+ 0.055)/1.055)**(2.4))
  srgb_img[srgb_img > 1.0] = 1.0
  srgb_img[srgb_img< 0.0] = 0
  return srgb_img



