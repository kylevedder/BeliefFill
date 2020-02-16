#!/usr/bin/env python3
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Convert network output to images.')
parser.add_argument('xs', type=str, help='input images')
parser.add_argument('ys', type=str, help='true output images')
parser.add_argument('yhats', type=str, help='network output images')
args = parser.parse_args()

def compress_images(images):
  if len(images.shape) == 4 and images.shape[1] == 1:
    images = np.squeeze(images, 1)
  assert(len(images.shape) == 3)
  return images

xs = compress_images(joblib.load(args.xs))
ys = compress_images(joblib.load(args.ys))
yhats = compress_images(joblib.load(args.yhats))

assert(xs.shape == ys.shape)
assert(yhats.shape == ys.shape)

for i in range(yhats.shape[0]):
  x = xs[i]
  y = ys[i]
  yhat = yhats[i]

  plt.subplot(231)
  plt.title("$x$")
  plt.imshow(x, cmap='gray')
  plt.clim(0, 1)

  plt.subplot(232)
  plt.title("$y$")
  plt.imshow(y, cmap='gray')
  plt.clim(0, 1)

  plt.subplot(233)
  plt.title("$\hat{y}$")
  plt.imshow(yhat, cmap='gray')
  plt.clim(0, 1)

  plt.subplot(234)
  plt.title("$y - x$")
  plt.imshow(x - y)
  plt.clim(-1, 1)

  plt.subplot(235)
  plt.title("$\hat{y} - x$")
  plt.imshow(yhat - x)
  plt.clim(-1, 1)

  plt.subplot(236)
  plt.title("$\hat{y} - y$")
  plt.imshow(yhat - y)
  plt.clim(-1, 1)


  plt.savefig("tmp/img{0:05d}.png".format(i))
  plt.clf()
  


# for i in range(100):
#   raw = occupancy_lst[i][0]
#   ablated = ablated_occupancy_lst[i][0]
#   diff = raw - ablated
#   plt.subplot(131)
#   plt.imshow(raw)
#   plt.subplot(132)
#   plt.imshow(ablated)
#   plt.subplot(133)
#   plt.imshow(diff)
#   plt.savefig("img{0:05d}.png".format(i))
#   plt.clf()