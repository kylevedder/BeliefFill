#!/usr/bin/env python3
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scan_data import ScanData
import cv2

parser = argparse.ArgumentParser(description='Convert laser scans to occupancy grids.')
parser.add_argument('scans', type=str, help='scan data file')
args = parser.parse_args()

message_lst = joblib.load(args.scans)

def scan_to_cartesian_points(scan):
  points = []
  kMinThreshold = 0.15
  for idx, r in enumerate(scan.ranges):
    if not np.isfinite(r) or r < scan.range_min + kMinThreshold  or r > scan.range_max:
      continue
    angle = scan.angle_min + idx * scan.angle_increment
    x = np.cos(angle) * r
    y = np.sin(angle) * r
    points.append((x, y))
  return points
    
points_lst = [scan_to_cartesian_points(m) for m in message_lst]
print([len(p) for p in points_lst][:10])


def points_to_occupancy(points, x_min, x_max, x_num_cells, y_min, y_max, y_num_cells):
  occupancy = np.zeros((x_num_cells, y_num_cells))

  def xy_to_cell_idx(x, y):
    if x < x_min or x > x_max or y < y_min or y > y_max:
      return None
    x_box_dist = (x_max - x_min) / x_num_cells
    y_box_dist = (y_max - y_min) / y_num_cells
    x_idx = int((x - x_min) / x_box_dist)
    y_idx = int((y - y_min) / y_box_dist)
    return x_idx, y_idx

  for x, y in points:
    idxs = xy_to_cell_idx(x, y)
    if idxs is None:
      continue
    x_idx, y_idx = idxs
    assert(x_idx >= 0)
    assert(y_idx >= 0)
    occupancy[x_idx, y_idx] = 1

  return occupancy


x_min = -1.5
x_max = 8.5
x_num_cells = 100
y_min = -5
y_max = 5
y_num_cells = 100

occupancy_lst = [points_to_occupancy(p, x_min, x_max, x_num_cells, y_min, y_max, y_num_cells) for p in points_lst]
occupancy_lst = [np.expand_dims(o, 0) for o in occupancy_lst]
occupancy_stack = np.concatenate(occupancy_lst, 0)

kBlurKernel = np.loadtxt('gaussian_blur.matrix', delimiter=',')

def blur_image(img):
  blurred_image = cv2.filter2D(img, -1, kBlurKernel)
  return blurred_image / np.amax(blurred_image)
  # return cv2.GaussianBlur(img, (5,5), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_REPLICATE)

blurred_occupancy_lst = [blur_image(o) for o in occupancy_lst]
blurred_occupancy_stack = np.concatenate(blurred_occupancy_lst, 0)

for i in range(500):

  # plt.imshow(occupancy_stack[i + 100], cmap='gray')
  # plt.colorbar()
  # name = 'img{0:05d}_occ.png'.format(i + 100)
  # plt.title(name)
  # plt.savefig(name)
  # plt.clf()
  name = 'img{0:05d}_raw_blur.png'.format(i)
  plt.subplot(121)
  plt.title("{0:05d} raw".format(i))
  plt.imshow(occupancy_stack[i], cmap='gray')
  plt.clim(0, 1)
  plt.subplot(122)
  plt.title("{0:05d} blur".format(i))
  plt.imshow(blurred_occupancy_stack[i], cmap='gray')
  plt.clim(0, 1)
  plt.savefig(name)
  plt.clf()
  print(i)
