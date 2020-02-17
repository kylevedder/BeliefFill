#!/usr/bin/env python3
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scan_data import ScanData
import cv2
import os
from PIL import Image


parser = argparse.ArgumentParser(description='Convert laser scans to occupancy grids.')
parser.add_argument('scans', type=str, help='scan data file')
args = parser.parse_args()

message_lst = joblib.load(args.scans)

def ablate_scan_region(scan, min_idx, max_idx):
  return scan._replace(ranges = (e if idx < min_idx or idx > max_idx else scan.range_max + 1 for idx, e in enumerate(scan.ranges) ) )

def polar_to_cartesian(angle, r):
  return np.cos(angle) * r, np.sin(angle) * r

def scan_to_cartesian_points(scan):
  points = []
  kMinThreshold = 0.15
  for idx, r in enumerate(scan.ranges):
    if not np.isfinite(r) or r < scan.range_min + kMinThreshold  or r > scan.range_max:
      continue
    angle = scan.angle_min + idx * scan.angle_increment
    points.append(polar_to_cartesian(angle, r))
  return points

x_min = -1.5
x_max = 8.5
x_num_cells = 100
y_min = -5
y_max = 5
y_num_cells = 100

x_box_dist = (x_max - x_min) / x_num_cells
y_box_dist = (y_max - y_min) / y_num_cells
assert(x_box_dist == y_box_dist)

def xy_to_cell_idx(x, y):
  if x < x_min or x > x_max or y < y_min or y > y_max:
    return None
  x_idx = int((x - x_min) / x_box_dist)
  y_idx = int((y - y_min) / y_box_dist)
  return x_idx, y_idx

def make_empty_occupancy():
  return np.zeros((x_num_cells, y_num_cells))

def make_indexed_occupancy():
  xs, ys = np.mgrid[0:x_num_cells,0:y_num_cells]
  return xs, ys

indexed_xs, indexed_ys = make_indexed_occupancy()
positions_xs = indexed_xs * x_box_dist + x_min + x_box_dist / 2
positions_ys = indexed_ys * y_box_dist + y_min + y_box_dist / 2

def points_to_occupancy(points):
  occupancy = make_empty_occupancy()

  for x, y in points:
    idxs = xy_to_cell_idx(x, y)
    if idxs is None:
      continue
    x_idx, y_idx = idxs
    assert(x_idx >= 0)
    assert(y_idx >= 0)
    occupancy[x_idx, y_idx] = 1

  return occupancy

def ablated_indices_to_mask(scan):
  min_x, min_y = polar_to_cartesian(scan.angle_min + ablate_index_min * scan.angle_increment, 1)
  max_x, max_y = polar_to_cartesian(scan.angle_min + ablate_index_max * scan.angle_increment, 1)

  min_cross = np.sign(positions_xs * min_y - min_x * positions_ys)
  max_cross = np.sign(positions_xs * max_y - max_x * positions_ys)
  return np.logical_and((min_cross != max_cross), (positions_xs > 0)).astype(np.float32)

kBlurKernel = np.loadtxt('gaussian_blur.matrix', delimiter=',')

def blur_image(img):
  blurred_image = cv2.filter2D(img, -1, kBlurKernel)
  return blurred_image / np.amax(blurred_image)

points_lst = [scan_to_cartesian_points(m) for m in message_lst]
occupancy_lst = [points_to_occupancy(p) for p in points_lst]

print("Ground truth occupancy complete")

ablate_index_min = 200
ablate_index_max = 250

masks_lst = [ablated_indices_to_mask(m) for m in message_lst]
# masks_lst = [ablated_indices_to_mask(p, ablate_index_min, ablate_index_max) for p in points_lst]

print("Mask occupancy complete")

ablated_message_lst = [ablate_scan_region(m, ablate_index_min, ablate_index_max) for m in message_lst]
ablated_points_lst = [scan_to_cartesian_points(m) for m in ablated_message_lst]
ablated_occupancy_lst = [points_to_occupancy(p) for p in ablated_points_lst]

print("Ablated occupancy complete")

blurred_occupancy_lst = [blur_image(i) for i in occupancy_lst]
ablated_blurred_occupancy_lst = [blur_image(i) for i in ablated_occupancy_lst]

os.makedirs("masks/", exist_ok=True)
os.makedirs("ablated/", exist_ok=True)
os.makedirs("true/", exist_ok=True)

def save_img(name, img):
  Image.fromarray(img * 255).convert('L').save(name)

def save_masks(idx, img):
  save_img("masks/{0:05d}.png".format(idx), img)

def save_ablated(idx, img):
  save_img("ablated/{0:05d}.png".format(idx), img)

def save_true(idx, img):
  save_img("true/{0:05d}.png".format(idx), img)

for idx, img in enumerate(masks_lst):
  save_masks(idx, img)

for idx, img in enumerate(blurred_occupancy_lst):
  save_true(idx, img)

for idx, img in enumerate(ablated_blurred_occupancy_lst):
  save_ablated(idx, img)

# blurred_occupancy_stack = make_stack(blurred_occupancy_lst)
# blurred_ablated_occupancy_stack = make_stack(ablated_blurred_occupancy_lst)

# joblib.dump(blurred_occupancy_stack, args.true)
# joblib.dump(blurred_ablated_occupancy_stack, args.ablated)

# for i in range(100):
#   raw = blurred_occupancy_lst[i]
#   ablated = ablated_blurred_occupancy_lst[i]
#   diff = raw - ablated
#   plt.subplot(131)
#   plt.imshow(raw, cmap='gray')
#   plt.clim(0, 1)
#   plt.subplot(132)
#   plt.imshow(ablated, cmap='gray')
#   plt.clim(0, 1)
#   plt.subplot(133)
#   plt.imshow(diff, cmap='gray')
#   plt.clim(0, 1)
#   plt.savefig("img{0:05d}.png".format(i))
#   plt.clf()
