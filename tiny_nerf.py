# -*- coding: utf-8 -*-
"""Copy of tiny_nerf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YR3AJPOzXI8v2S8TaIigDHJIEPB8frhn

##Tiny NeRF
This is a simplied version of the method presented in *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*

[Project Website](http://www.matthewtancik.com/nerf)

[arXiv Paper](https://arxiv.org/abs/2003.08934)

[Full Code](github.com/bmild/nerf)

Components not included in the notebook
*   5D input including view directions
*   Hierarchical Sampling
"""

# Commented out IPython magic to ensure Python compatibility.
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
     tensorflow_version =2

import os, sys

import wget

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt

# if not os.path.exists('tiny_nerf_data.npz'):
#     wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz

import requests
from img_utility import *
from pseudo_utility import *
SCALE_FACTOR = 1
AUTO_BRIGHT_OFF = False
AUTO_BRIGHT_THR = 0.001
SAMPLES = 8000

#from raw_utility import *

from file_utility import get_filenames
fname = 'tiny_nerf_data.npz'
url = 'http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/' + fname
r = requests.get(url)
open(fname , 'wb').write(r.content)

"""# Load Input Images and Poses"""

data = np.load('tiny_nerf_data.npz')
images = data['images']
log_images = []
#images = np.array(logify_image(images)) #logify crude
for img in images:
    img1 = srgb_to_xyz(img, max_val=255)
    img1 = float_to_tiff(img1)
    img1 = bgr_to_rgb(img1)
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
print(testimg)

# max_val = 255/max_val
# print(max_val)
# new_image = (np.floor_divide((testimg * max_val), 1)).astype(np.uint8)
# plt.imshow(new_image)
# plt.title('new image')
# plt.show()
logdiv = 11.4
displog_image = (testimg / logdiv) * 255
displog_image = displog_image.astype("uint8")
displog_image = cv2.cvtColor(displog_image, cv2.COLOR_BGR2RGB)
plt.imshow(displog_image)
plt.title('log view new')
plt.show()

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

def posenc(x):
  rets = [x]
  for i in range(L_embed):
    for fn in [tf.sin, tf.cos]:
      rets.append(fn(2.**i * x))
  return tf.concat(rets, -1)

L_embed = 6
embed_fn = posenc
# L_embed = 0
# embed_fn = tf.identity

def init_model(D=8, W=256):
    relu = tf.keras.layers.ReLU()
    dense = lambda W=W, act=relu : tf.keras.layers.Dense(W, activation=act)

    inputs = tf.keras.Input(shape=(3 + 3*2*L_embed))
    outputs = inputs
    for i in range(D):
        outputs = dense()(outputs)
        if i%4==0 and i>0:
            outputs = tf.concat([outputs, inputs], -1)
    outputs = dense(4, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_rays(H, W, focal, c2w):
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d))
    return rays_o, rays_d



def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):

    def batchify(fn, chunk=1024*32):
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    # Compute 3D query points
    z_vals = tf.linspace(near, far, N_samples)
    if rand:
      z_vals += tf.random.uniform(list(rays_o.shape[:-1]) + [N_samples]) * (far-near)/N_samples
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    # Run network
    pts_flat = tf.reshape(pts, [-1,3])
    pts_flat = embed_fn(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    raw = tf.reshape(raw, list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    sigma_a = tf.nn.relu(raw[...,3])
    rgb = tf.math.exp(raw[...,:3])
    #rgb = processRAW_img(rgb)
    # for r in raw[...,:3]:
    #     if r.>1:
    #         r=1
#########
    # Compute opacities and colors
    # sigma_a = tf.nn.relu(raw[...,3])
    # #rgb = tf.math.sigmoid(raw[...,:3])
    # rgb = e**raw , normalize to [0 1], clip anything over 1
    # then convert to srgb
########
    # Do volume rendering
    dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], tf.broadcast_to([1e10], z_vals[...,:1].shape)], -1)
    alpha = 1.-tf.exp(-sigma_a * dists)
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    rgb_map = tf.reduce_sum(weights[...,None] * rgb, -2)
    depth_map = tf.reduce_sum(weights * z_vals, -1)
    acc_map = tf.reduce_sum(weights, -1)

    return rgb_map, depth_map, acc_map

"""Here we optimize the model. We plot a rendered holdout view and its PSNR every 50 iterations."""

model = init_model()
optimizer = tf.keras.optimizers.Adam(5e-4)

N_samples = 64
N_iters = 250
psnrs = []
iternums = []
i_plot = 250

import time
t = time.time()
for i in range(N_iters+1):

    img_i = np.random.randint(images.shape[0])
    target = images[img_i]
    pose = poses[img_i]
    rays_o, rays_d = get_rays(H, W, focal, pose)
    with tf.GradientTape() as tape:
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples, rand=True)
        loss = tf.reduce_mean(tf.square((rgb - target)))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if i%i_plot==0:
        print(i, (time.time() - t) / i_plot, 'secs per iter')
        t = time.time()

        # Render the holdout view for logging
        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        loss = tf.reduce_mean(tf.square((rgb - target)))
        psnr = -10. * tf.math.log(loss) / tf.math.log(10.)

        psnrs.append(psnr.numpy())
        iternums.append(i)

        plt.figure(figsize=(10,4))
        plt.subplot(131)
        plt.imshow(rgb)
        # icecream float
        max_val = np.max(rgb)
        max_val = 255 / max_val
        new_image = (np.floor_divide((rgb * max_val), 1)).astype(np.uint8)
        plt.subplot(132)
        plt.imshow(new_image)
        plt.title('new image')
        #plt.show()
        plt.title(f'Iteration: {i}')
        plt.subplot(133)
        plt.plot(iternums, psnrs)
        plt.title('PSNR')
        plt.show()

print('Done')

"""# Interactive Visualization"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from ipywidgets import interactive, widgets


trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def f(**kwargs):
    c2w = pose_spherical(**kwargs)
    rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
    rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
    img = np.clip(rgb,0,1)

    plt.figure(2, figsize=(20,6))
    plt.imshow(img)
    plt.show()


sldr = lambda v, mi, ma: widgets.FloatSlider(
    value=v,
    min=mi,
    max=ma,
    step=.01,
)

names = [
    ['theta', [100., 0., 360]],
    ['phi', [-30., -90, 0]],
    ['radius', [4., 3., 5.]],
]

interactive_plot = interactive(f, **{s[0] : sldr(*s[1]) for s in names})
output = interactive_plot.children[-1]
output.layout.height = '350px'
interactive_plot

"""# Render 360 Video"""

frames = []
for th in tqdm(np.linspace(0., 360., 120, endpoint=False)):
    c2w = pose_spherical(th, -30., 4.)
    rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
    rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
    frames.append((255*np.clip(rgb,0,1)).astype(np.uint8))

import imageio
f = 'video.mp4'
imageio.mimwrite(f, frames, fps=30, quality=7)

import imageio.v2 as iio
w = iio.get_writer('my_video.mp4', mode='I',fps=1)

from IPython.display import HTML
from base64 import b64encode
mp4 = open('video.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls autoplay loop>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)



