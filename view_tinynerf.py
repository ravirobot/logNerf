import matplotlib.pyplot as plt
import numpy as np
import requests
from img_utility import *
from pseudo_utility import *

data = np.load('tiny_nerf_data.npz')
images = data['images']
log_images = []
#images = np.array(logify_image(images)) #logify crude
for img in images:
    img1 = srgb_to_xyz(img, max_val=255)
    img1 = float_to_tiff(img1)
    #img1 = bgr_to_rgb(img1)
    img1 = tiff_to_log(img1)
    log_images.append(img1)
images = np.array(log_images)
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]

print(images.shape, poses.shape, focal)


testimg, testpose = images[101], poses[101]
images = images[:100,...,:3]
poses = poses[:100]

plt.imshow(testimg)
plt.show()

print(testimg.shape)
print(testimg[50])
"""# Optimize NeRF"""
#Introducing, Heather's code

# img = testimg
# img = srgb_to_xyz(img, max_val=1)
# img = float_to_tiff(img)
# img = bgr_to_rgb(img)
# #img = float_to_tiff(img)
# #img = srgb_to_xyz(img, max_val=1)
# max_16_bit_val = 65535
# sixteen_bit_image = img# (np.floor_divide((img * max_16_bit_val), 1)).astype(np.uint16)
# plt.imshow(img)
# plt.title('linear')
# plt.show()
#
# #logify
# #def tiff_to_log(sixteen_bit_image):
# # sixteen_bit_image = sixteen_bit_image.astype(np.float32)
# # sixteen_bit_image[sixteen_bit_image!=0] = np.log(sixteen_bit_image[sixteen_bit_image!=0])
# #return sixteen_bit_image
# img = tiff_to_log(sixteen_bit_image)
# plt.imshow(sixteen_bit_image)
# plt.title('log image')
# plt.show()
#Introducing, Heather's code
#icecream float
max_val = np.max(testimg)
# max_val = 255/max_val
# print(max_val)
# new_image = (np.floor_divide((testimg * max_val), 1)).astype(np.uint8)
# plt.imshow(new_image)
# plt.title('new image')
# plt.show()
logdiv = 11.4
displog_image = (testimg / logdiv) * 255
displog_image = displog_image.astype("uint8")
#displog_image = cv2.cvtColor(displog_image, cv2.COLOR_BGR2RGB)
plt.imshow(displog_image)
plt.title('log view new')
plt.show()
