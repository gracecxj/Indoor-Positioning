# import numpy as np
# import re
import os
import xml.dom.minidom
# import collections


class LocationFile(object):
    def __init__(self, file_name):
        # self.north_west = (55.94510000000000, -3.188000000000000)
        self.north_west = (55.945139, -3.18781)
        # self.south_east = (55.94450000000000, -3.186000000000000)
        self.south_east = (55.944600, -3.186537)
        self.num_grid_y = 60  # latitude
        self.num_grid_x = 80  # longitude
        self.delta_lat = 0
        self.delta_lng = 0

        self.num_loc = 0
        self.raw_dict = {}
        self.grid_dict = {}

        self.fn = file_name
        self.parse_file(file_name)
        self.calculate_grid()

    def parse_file(self, file_name):
        dom = xml.dom.minidom.parse(file_name)
        root = dom.documentElement
        loc_list = root.getElementsByTagName('loc')

        self.num_loc = loc_list.length

        i = 0
        for item in loc_list:
            i = i + 1
            try:
                lat = float(item.getAttribute("lat"))
                lng = float(item.getAttribute("lng"))
            except ValueError:
                print('invalid input %d: %s,%s'.format(i, lat, lng))
            # print(i, item.getAttribute("t"), lat, lng)
            self.raw_dict[(i, item.getAttribute("t"))] = (lat, lng)

        # self.ordered_raw_dict = collections.OrderedDict(sorted(self.raw_dict.items(), key=lambda t: t[1]))

    def calculate_grid(self):
        max_lat = abs(self.north_west[0] - self.south_east[0])  # 0.0006
        max_lng = abs(self.north_west[1] - self.south_east[1])  # 0.002
        self.delta_lat = max_lat / self.num_grid_y      # 3e-06
        self.delta_lng = max_lng / self.num_grid_x      # 1e-05

        for k, v in self.raw_dict.items():
            y = abs(v[0] - self.north_west[0]) // self.delta_lat       # index start from 0
            x = abs(v[1] - self.north_west[1]) // self.delta_lng       # index start from 0
            index = y * self.num_grid_x + x         # corresponding index from 0 to 4800
            self.grid_dict[k] = (int(x), int(y), int(index))
            print(k, "-->", self.grid_dict[k])

    # write formation 1
    def write_to_file(self):
        filename = os.path.basename(self.fn)
        w_filename = "./foreground_results/grid_"+os.path.splitext(filename)[0] + ".txt"
        print("ppp_filename:", w_filename)
        w_file = open(w_filename, 'w')

        for key, value in self.grid_dict.items():
            # w_file.write(str(key) + "\t" + str(value[0]) + "\t" + str(value[1]) + "\t" + str(value[2]) + "\n")
            w_file.write(str(value[0]) + "\t" + str(value[1]) + "\t" + str(value[2]) + "\n")
        w_file.close()

    # write formation 2（simple）
    def write_latlng(self):
        filename = os.path.basename(self.fn)
        w_filename = "./foreground_results/latlng_"+os.path.splitext(filename)[0] + ".txt"
        w_file = open(w_filename, 'w')
        for key, value in self.raw_dict.items():
            w_file.write(str(value[1]) + "\t" + str(value[0]) + "\n")
        w_file.close()


def gci(path, loc_file_list):
    dirs = os.listdir(path)
    for dir in dirs:
        fi_d = os.path.join(path, dir)
        if os.path.isdir(fi_d):
            gci(fi_d, loc_file_list)
        else:
            loc_file_list.append(LocationFile(os.path.join(path, fi_d)))
            # loc_file_list.append(LocationFile(fi_d))  # same as previous line


loc_file = []
gci("/Users/chenxingji/PycharmProjects/InLoc_preprocessing/foreground", loc_file)


for file in loc_file:
    file.write_to_file()
    file.write_latlng()
