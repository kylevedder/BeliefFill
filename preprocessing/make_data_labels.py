#!/usr/bin/env python3
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scan_data import ScanData
import cv2

parser = argparse.ArgumentParser(description='Convert laser scans to occupancy grids.')
parser.add_argument('scans', type=str, help='scan data file')
parser.add_argument('true', type=str, help='true occupancy grid')
parser.add_argument('ablated', type=str, help='ablated occupancy grid')
args = parser.parse_args()

message_lst = joblib.load(args.scans)


def ablate_scan_region(scan, min_idx, max_idx):
  return scan._replace(ranges = (e if idx < min_idx or idx > max_idx else scan.range_max + 1 for idx, e in enumerate(scan.ranges) ) )

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

def points_to_occupancy(points):
  x_min = -1.5
  x_max = 8.5
  x_num_cells = 100
  y_min = -5
  y_max = 5
  y_num_cells = 100
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

kBlurKernel = np.loadtxt('gaussian_blur.matrix', delimiter=',')

def blur_image(img):
  blurred_image = cv2.filter2D(img, -1, kBlurKernel)
  return blurred_image / np.amax(blurred_image)

def make_stack(lst):
  return np.concatenate([np.expand_dims(e, 0) for e in lst])

points_lst = [scan_to_cartesian_points(m) for m in message_lst]
occupancy_lst = [points_to_occupancy(p) for p in points_lst]

ablated_message_lst = [ablate_scan_region(m, 200, 250) for m in message_lst]
ablated_points_lst = [scan_to_cartesian_points(m) for m in ablated_message_lst]
ablated_occupancy_lst = [points_to_occupancy(p) for p in ablated_points_lst]

blurred_occupancy_lst = [blur_image(i) for i in occupancy_lst]
ablated_blurred_occupancy_lst = [blur_image(i) for i in ablated_occupancy_lst]

occupancy_stack = make_stack(blurred_occupancy_lst)
ablated_occupancy_stack = make_stack(ablated_blurred_occupancy_lst)

joblib.dump(occupancy_stack, args.true)
joblib.dump(ablated_occupancy_stack, args.ablated)

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
