#!/usr/bin/env python2
import rosbag
import argparse
import joblib
from scan_data import ScanData

parser = argparse.ArgumentParser(description='Convert .bag file into processable data.')
parser.add_argument('bag', type=str, help='.bag file')
parser.add_argument('output', type=str, help='output data file')

args = parser.parse_args()



bag = rosbag.Bag(args.bag)
messages = [msg for _, msg, _ in bag.read_messages(topics=['/scan'])]
bag.close()

def msg_to_scan_data(msg):
  return ScanData(angle_min=msg.angle_min, angle_max=msg.angle_max, angle_increment=msg.angle_increment, range_min=msg.range_min, range_max=msg.range_max, ranges=msg.ranges)

messages = [msg_to_scan_data(m) for m in messages]

joblib.dump(messages, args.output)