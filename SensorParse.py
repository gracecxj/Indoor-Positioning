import os
import re
import xml.dom.minidom
import collections
import numpy as np
import pickle
import h5py

# define some constant variables
north_west = (55.945139, -3.18781)
south_east = (55.944600, -3.186537)
num_grid_y = 30  # latitude
num_grid_x = 40  # longitude
max_lat = abs(north_west[0] - south_east[0])  # 0.0006 # 0.0005393
max_lng = abs(north_west[1] - south_east[1])  # 0.002  # 0.001280
delta_lat = max_lat / num_grid_y  # 3e-06
delta_lng = max_lng / num_grid_x  # 1e-05

# ROW_SIZE = 2000
NUM_COLUMNS = 3*100 + 3*100 + 1*102

# *****************************************************************************************************
# 2. Read the distinct access point id from file("wifi_filename") into dictionary

wifi_filename = "./preprocessing/wifi_id.txt"


def read_ap_to_dict(filename):
    ap_dict = collections.OrderedDict()
    with open(filename) as file:
        for line in file:
            elements = re.split(r'[\s]', line.strip())
            ap_dict[elements[0]] = (elements[1], elements[2])
    return ap_dict


wifi_dict = read_ap_to_dict(wifi_filename)


# *****************************************************************************************************
# 3. Pre-processing the background files, generate standard input data
# instantiate a SensorFile object for each background file collected

# Each time the sensor signal for one location is processed, this function returns a single sample's input
# corresponding to the incoming parameters （2秒即2000毫秒）此函数根据传入的参数计算，并返回一个样本的标准输入数据格式（100个acc, 100个mag, 1个wr,
# 但一个WiFi record包括102个值，这是由"wifi_id.txt"有几个唯一的AP确定的）
def reduce_frequency_average(t_start, t_end, raw_acc_list, raw_mag_list):
    sequence1 = np.arange(t_start, t_end + 1, 20)  # t_end need to plus 1 is because the last number is not included
    # sequence2 = np.arange(t_start, t_end + 1, 500)
    # the following three list is for one location, we need reduce the frequency of them
    # output: acc_reduced, mag_reduced will both have 100 values respectively,
    #         and wr_reduced will have only one ap list
    acc_reduced = []
    mag_reduced = []

    acc_mark = False
    mag_mark = False

    # acc/mag, each time cell ( = 20ms)
    for i in range(len(sequence1) - 1):
        t1 = sequence1[i]
        t2 = sequence1[i + 1]

        cell_acc = [acc[1] for acc in raw_acc_list if t1 < acc[0][0] < t2]
        cell_mag = [mag[1] for mag in raw_mag_list if t1 < mag[0][0] < t2]

        # from a list of element into one element(in a single cell)
        if len(cell_acc) > 1:
            cell_acc = reduce_cell_average(cell_acc, 1)
            if acc_mark:    # if the previous cell do not have data , which means acc_mark has been marked "True" in
                # the previous round
                acc_mark = False
                if i == 1:
                    acc_reduced[0] = cell_acc
                else:
                    # acc_reduced[i - 1] = (cell_acc + acc_reduced[i - 2]) / 2
                    acc_reduced[i - 1] = reduce_cell_average([cell_acc, mag_reduced[i-2]], 1)
        elif len(cell_acc) == 1:
            cell_acc = cell_acc[0]
            if acc_mark:    # if the previous cell do not have data, which means acc_mark has been marked "True" in
                # the previous round
                acc_mark = False
                if i == 1:
                    acc_reduced[0] = cell_acc
                else:
                    # acc_reduced[i - 1] = (cell_acc + acc_reduced[i - 2]) / 2
                    acc_reduced[i - 1] = reduce_cell_average([cell_acc, mag_reduced[i-2]], 1)
        elif len(cell_acc) == 0:
            acc_mark = True
            cell_acc = (0, 0, 0)
            # print(t_start, t_end, i, ": null acc")

        if len(cell_mag) > 1:
            cell_mag = reduce_cell_average(cell_mag, 1)
            if mag_mark:
                mag_mark = False
                if i == 1:
                    mag_reduced[0] = cell_mag
                else:
                    # mag_reduced[i - 1] = (cell_mag + mag_reduced[i - 2]) / 2
                    mag_reduced[i - 1] = reduce_cell_average([cell_mag, mag_reduced[i-2]], 1)
        elif len(cell_mag) == 1:
            cell_mag = cell_mag[0]
            if mag_mark:
                mag_mark = False
                if i == 1:
                    mag_reduced[0] = cell_mag
                else:
                    # mag_reduced[i - 1] = (cell_mag + mag_reduced[i - 2]) / 2
                    mag_reduced[i - 1] = reduce_cell_average([cell_mag, mag_reduced[i-2]], 1)
        elif len(cell_mag) == 0:
            mag_mark = True
            cell_mag = (0, 0, 0)
            # print(t_start, t_end, i, ": null mag")

        acc_reduced.append(cell_acc)
        mag_reduced.append(cell_mag)

    # # wr, each time cell ( = 500ms)
    # for i in range(len(sequence2) - 1):
    #     t1 = sequence2[i]
    #     t2 = sequence2[i + 1]
    #
    #     cell_wr = [wr[1] for wr in raw_wr_list if t1 < wr[0][1] < t2]
    #
    #     # from a list of element into one element(in a single cell)
    #     if len(cell_wr) > 1:
    #         cell_wr = reduce_cell_average(cell_wr, 2)
    #     else:
    #         cell_wr = reduce_cell_average(cell_wr, 3)
    #     wr_reduced.append(cell_wr)

    return acc_reduced, mag_reduced


# Input: "raw_list" is a sequence of sensor data in a cell(20ms for acc/mag, 500ms for wr)
# Output: "reduced" is a single sensor data  calculated by averaging the input list
# ** The following function reduces a list of elements into one element **
# 计算每一个cell(对acc/mag是20ms, 对wr是500ms)的reduce之后的值并返回, 该函数的输入参数raw_list去掉了时间信息，只有传感器的值
# 返回值：传入的是acc/mag返回值为3个，传入的是wr返回值为102个
def reduce_cell_average(raw_list, mark):
    # How many values are there for each element?
    # acc, mag will have 3 values, while wr will have approximately 50 values

    # acc/mag
    if mark == 1:
        element = np.array(raw_list)
        element = np.mean(element, axis=0)
    # # wr
    # elif mark == 2:
    #     wr_num = len(raw_list)
    #     ap_num = len(SensorFile.world_ap_dict)  # standard input need same number of input ap
    #     element = np.zeros((wr_num, ap_num))
    #     for i, ele in zip(range(wr_num), raw_list):
    #         for ap in ele:
    #             ap_id = ap[0]
    #             ap_val = ap[1]
    #             # find out the index（colum index in element） of this ap_id
    #             ap_index = int(SensorFile.world_ap_dict[ap_id][1]) - 1
    #             element[i, ap_index] = ap_val
    #     element = np.mean(element, axis=0)
    # # only 1 wr in this cell
    # elif mark == 3:
    #     ap_num = len(SensorFile.world_ap_dict)  # standard input need same number of input ap
    #     element = np.zeros(ap_num)
    #     if raw_list:
    #         for ap in raw_list[0]:
    #             ap_id = ap[0]
    #             ap_val = ap[1]
    #             # find out the index（colum index in element） of this ap_id
    #             ap_index = int(SensorFile.world_ap_dict[ap_id][1]) - 1
    #             element[ap_index] = ap_val
    return element


class SensorFile(object):
    # Class variable
    world_ap_dict = wifi_dict
    file_rank = 0

    def __init__(self, file_name):
        # Member variables
        self.acc_dict = collections.OrderedDict()
        self.mag_dict = collections.OrderedDict()
        self.wr_dict = collections.OrderedDict()
        self.loc_dict = collections.OrderedDict()
        self.fn = file_name

        # Transfer the data from raw file into internal data structure
        self.first_parse_file(file_name)
        self.sample_num = len(self.loc_dict)
        self.f_inputs = np.zeros((self.sample_num, NUM_COLUMNS))
        self.f_outputs = np.zeros((self.sample_num, 5))

        # Filter out(reduce frequency) useful input data according to recorded location
        self.threshold_and_filter()

        # Save standard input and output into files
        # self.save_txt_and_pickle()
        self.save_overall_txt()
        self.save_overall_hdf5()

    def first_parse_file(self, file_name):
        dom = xml.dom.minidom.parse(file_name)
        root = dom.documentElement

        acc_list = root.getElementsByTagName('a')
        mag_list = root.getElementsByTagName('m')
        wr_list = root.getElementsByTagName('wr')
        loc_list = root.getElementsByTagName('loc')

        print("# accelerometer:", acc_list.length)
        print("# magnetometer:", mag_list.length)
        print("# wifi record:", wr_list.length)
        print("# loc record:", loc_list.length)

        # accelerometer
        for item in acc_list:
            try:
                t = int(item.getAttribute("t"))
                x = float(item.getAttribute("x"))
                y = float(item.getAttribute("y"))
                z = float(item.getAttribute("z"))
                st = int(item.getAttribute("st"))
            except ValueError:
                print('invalid input accelerometer: %s,%s,%s,%s'.format(x, y, z, st))
            self.acc_dict[t, st] = (x, y, z)

        # magnetometer
        for item in mag_list:
            try:
                t = int(item.getAttribute("t"))
                x = float(item.getAttribute("x"))
                y = float(item.getAttribute("y"))
                z = float(item.getAttribute("z"))
                st = int(item.getAttribute("st"))
            except ValueError:
                print('invalid input magnetometer: %s,%s,%s,%s'.format(x, y, z, st))
            self.mag_dict[(t, st)] = (x, y, z)

        # location(user input)
        for item, i in zip(loc_list, range(len(loc_list))):
            try:
                t = int(item.getAttribute("t"))
                lat = float(item.getAttribute("lat"))
                lng = float(item.getAttribute("lng"))
            except ValueError:
                print('invalid input %d: %s,%s'.format(i, lat, lng))
            self.loc_dict[(i, t)] = (lat, lng)

        # wifi record
        # for item, i in zip(wr_list, range(len(wr_list))):  # for each time step 有wr记录的time step
        for item in wr_list:  # for each time step 有wr记录的time step
            t = int(item.getAttribute("t"))
            # print(i, "->", t, len(item.childNodes)//2)

            # ap_list是一个ap的列表，一个ap_list表示一个<wr>，代表一个time step记录下来的一个ap的列表
            ap_list = list()
            for record, j in zip(item.childNodes, range(len(item.childNodes))):  # for each AP
                if j % 2:
                    ap = item.childNodes[j].getAttribute("b")
                    s = item.childNodes[j].getAttribute("s")
                    if ap not in self.world_ap_dict.keys():
                        # self.world_wifi[ap] = 1
                        print("{} not in world ap dict".format(ap))
                    else:
                        ap_list.append((ap, s))
            # self.wr_dict[(i, t)] = ap_list
            self.wr_dict[t] = ap_list

    # 每处理一个background文件调用一次该函数
    def threshold_and_filter(self):

        for k, v in self.loc_dict.items():
            # for each recorded location
            # 为每个loc筛选出符合时间条件的传感器数据
            time_start = k[1] - 2000
            time_end = k[1]

            # -------------Select 2 sec previous sensor data for acc/mag---------------
            acc_threshold = [(tt, vv) for tt, vv in self.acc_dict.items() if time_start < tt[0] < time_end]
            mag_threshold = [(tt, vv) for tt, vv in self.mag_dict.items() if time_start < tt[0] < time_end]
            (acc_reduced, mag_reduced) = reduce_frequency_average(time_start, time_end, acc_threshold, mag_threshold)

            # --------------Select closest wifi record for each location-----------------
            wr_reduced = self.select_closest_wr(time_end)

            # ------------------------------------------------------------------------------------------------
            # following is save standardised input/output data into object variables self.inputs/self.outputs
            # k[0] indicates the index of this sample in this background file

            # add acc(0:300) into f_inputs
            for acc, i in zip(acc_reduced, range(len(acc_reduced))):
                self.f_inputs[k[0], 3 * i:3 * (i + 1)] = acc

            # add mag(300:600) into f_inputs
            for mag, i in zip(mag_reduced, range(len(mag_reduced))):
                self.f_inputs[k[0], 300 + 3 * i: 300 + 3 * (i + 1)] = mag

            # add wr(600:702) into f_inputs
            self.f_inputs[k[0], 600:] = np.array(wr_reduced).reshape(1, len(SensorFile.world_ap_dict.keys()))

            # add (lat, lng, x, y, index) into f_outputs
            self.f_outputs[k[0]] = np.array(SensorFile.latlng_to_grid(v[0], v[1]))

    def select_closest_wr(self, loc_time):
        try:
            wr_selected = self.wr_dict[loc_time]
        except KeyError:
            # pre_dis = time_end - next(iter(self.wr_dict))
            for wr_t in self.wr_dict.keys():  # self.wr_dict is an ordered dict
                cur_dis = loc_time - wr_t
                # 仍然没到time_end,扫描略过之前的所有wr,记住time_end的前一个wr
                if cur_dis > 0:
                    pre_dis = cur_dis
                    wr_selected = self.wr_dict[wr_t]
                    continue
                # 读到了time_end后的那个wr，比较前面一个wr和后面一个wr，哪个更近
                else:
                    # before "time_end" is the closest
                    if pre_dis + cur_dis < 0:
                        tt = loc_time - pre_dis
                    # after "time_end" is the closest
                    else:
                        tt = loc_time - cur_dis
                    wr_selected = self.wr_dict[tt]
                    break
        # formalise the selected wifi record
        return SensorFile.formalize_wr(wr_selected)

    @staticmethod
    def formalize_wr(wr):
        ap_num = len(SensorFile.world_ap_dict)  # standard input need same number of input ap
        element = np.zeros(ap_num)
        for ap in wr:
            ap_id = ap[0]
            ap_val = ap[1]
            # find out the index（column index in element） of this ap_id
            ap_index = int(SensorFile.world_ap_dict[ap_id][1]) - 1
            element[ap_index] = ap_val
        return element

    @staticmethod
    def latlng_to_grid(lat, lng):
        y = abs(lat - north_west[0]) // delta_lat  # index start from 0
        x = abs(lng - north_west[1]) // delta_lng  # index start from 0
        index = y * num_grid_x + x  # corresponding index from 0 to 4800
        return float(lat), float(lng), int(x), int(y), int(index)

    # save the standard input and output into ".txt" and ".pickle" files separately
    def save_txt_and_pickle(self):
        filename = os.path.basename(self.fn)
        t_filename = "./background_results/out_in_" + os.path.splitext(filename)[0] + ".txt"
        p_filename = "./pickle/" + os.path.splitext(filename)[0] + ".pickle"

        with open(p_filename, 'wb') as handle:
            pickle.dump(self.f_inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.f_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(t_filename, "wb") as f:
            write_text = np.hstack((self.f_outputs, self.f_inputs))
            np.savetxt(f, write_text, delimiter=",", newline='\n')

    # write the overall standard input and output into a single "out_in_overall.txt" file
    def save_overall_txt(self):
        txt_filename = "./background_results/out_in_overall(gridsize2).txt"
        write_text = np.hstack((self.f_outputs, self.f_inputs))
        with open(txt_filename, "ab") as f:
            np.savetxt(f, write_text, delimiter=",", newline='\n')

    # write the overall standard input and output into a single "out_in_overall.h5" file
    def save_overall_hdf5(self):
        h5_filename = "./background_results/out_in_overall(gridsize2).h5"
        h5_file = h5py.File(h5_filename, mode='a')
        write_content = np.hstack((self.f_outputs, self.f_inputs))
        h5_file.create_dataset(os.path.basename(self.fn), data=write_content)
        h5_file.close()


# Iterate over all the background file in the directory "background"
def iterate(path):
    dirs = os.listdir(path)
    for dir in dirs:
        if dir != ".DS_Store":
            fi_d = os.path.join(path, dir)
            if os.path.isdir(fi_d):
                iterate(fi_d)
            else:
                SensorFile(fi_d)
        else:
            pass
            # using "continue" here is the same as using "pass"


file1 = "./background_results/out_in_overall(gridsize2).h5"
file2 = "./background_results/out_in_overall(gridsize2).txt"
if os.path.isfile(file1):
    os.remove(file1)
if os.path.isfile(file2):
    os.remove(file2)

iterate("/Users/chenxingji/PycharmProjects/InLoc_preprocessing/background")
