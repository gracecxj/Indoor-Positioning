import os
import re
import xml.dom.minidom
import collections
import numpy as np
import pickle as pkl
import h5py
from preprocessing import Masking

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
# NUM_COLUMNS = 3*100 + 3*100 + 1*102     # accelerometer, magnetometer, wifi
NUM_COLUMNS = 1*102    # wr:(102 dimensional)
# loc: lat + lng + x + y + index(1 dimensional)

X_GRID_NUM = 40

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
# 3. getting the masking dict

def get_masking_dict(mask):

    # get distinct grid index
    exist_grid_list = []
    y_list, x_list = np.where(mask == 1)
    for i, j in zip(x_list, y_list):        # ith-column, jth-row
        index = j * X_GRID_NUM + i
        exist_grid_list.append(index)

    exist_grid_list = sorted(exist_grid_list)

    trasfer_dicts = {}
    for masked_ind, origin_ind in enumerate(exist_grid_list):
        trasfer_dicts[origin_ind] = masked_ind

    num_classes = len(exist_grid_list)

    return trasfer_dicts, num_classes


def generate_converting_dict():
    # define the masking area(using the polygon's vertex)
    vertices = [(55.944949749302125, -3.1877678632736206),
                (55.944643901703756, -3.1876143068075184),
                (55.944738904347155, -3.1869940459728237),
                (55.94483653527933, -3.187024220824241),
                (55.94482339266817, -3.187117762863636),
                (55.944765001870415, -3.1870899349451065),
                (55.94469421933837, -3.1875495985150337),
                (55.944866012262395, -3.187636099755764),
                (55.94498992799679, -3.186802938580513),
                (55.944825457935934, -3.186661116778851),
                (55.94484179232254, -3.1865846738219257),
                (55.94509957431598, -3.1867925450205803),
                (55.94508568079271, -3.1868706643581386),
                (55.94503949417953, -3.1868478655815125),
                (55.944909195063346, -3.187657222151756),
                (55.94496157761375, -3.18768136203289)]

    array = Masking.masking(vertices)  # Return the index of all reserved grids
    return get_masking_dict(array)


origin_to_masked_dict, NUM_CLASSES = generate_converting_dict()


# *****************************************************************************************************
# 4. Parsing file, get the interpolated wr(102* ap)+ loc(lat*1,lng*1, x*1, y*1, index*1)  ->  107 dimension / sample

X_GRID_NUM = 40
M_count = np.zeros((NUM_CLASSES, NUM_CLASSES))

# wr_container is a dict, the key is a tuple,
# the first element represents the start index, the second element represent the end index
wr_container = {}


# 根据wr和loc中每两个相邻的sample更新全局矩阵中的一个元素, wr和loc均有self.sample_num行
def update_global_matrix(wr, loc):

    for i in range(np.shape(loc)[0]-1):
        # convert original index into masked index
        index1 = origin_to_masked_dict[loc[i, -1]]
        index2 = origin_to_masked_dict[loc[i+1, -1]]

        M_count[index1, index2] += 1

        # wr_container是一个大的字典，key是（index1,index2）的转换，value是一个102个元素的list,每个元素表示一个ap的一个小字典
        # 若已经存在index1到index2的转换样本
        if (index1, index2) in wr_container.keys():
            # 遍历每一个AP
            for ap in range(NUM_COLUMNS):
                # 取出wr[i+1]的第ap个AP的强度作为strength
                strength = wr[i + 1][ap]
                if strength not in wr_container[(index1, index2)][ap].keys():
                    wr_container[(index1, index2)][ap][
                        strength] = 1  # wr_container[(index1, index2)][ap]是一个小字典，对应着第ap个AP
                else:
                    wr_container[(index1, index2)][ap][strength] += 1
        # 若还没有index1到index2的转换样本
        else:
            # small_dict_list is a fix length list, each element in this list is a sub dictionary
            small_dict_list = [{} for _ in range(NUM_COLUMNS)]
            wr_container[(index1, index2)] = small_dict_list    # 将初始化的这个102个元素的list赋值给该transfer
            # 遍历每一个AP
            for ap in range(NUM_COLUMNS):
                # 取出wr[i+1]的第ap个AP的强度作为strength
                strength = wr[i+1][ap]
                if strength not in wr_container[(index1, index2)][ap].keys():
                    wr_container[(index1, index2)][ap][strength] = 1         # wr_container[(index1, index2)][ap]是一个小字典，对应着第ap个AP
                else:
                    wr_container[(index1, index2)][ap][strength] += 1



class SensorFile(object):
    # Class variable
    world_ap_dict = wifi_dict
    file_rank = 0

    def __init__(self, file_name):
        # Member variables
        self.wr_dict = collections.OrderedDict()
        self.loc_dict = collections.OrderedDict()
        self.fn = file_name

        # Transfer the data from raw file into internal data structure
        self.first_parse_file(file_name)
        self.sample_num = len(self.new_wr_dict)     # 这里的sample_num和之前的类不一样了，这里将原来的loc_dict改成了new_wr_dict
        self.f_wr = np.zeros((self.sample_num, NUM_COLUMNS))
        self.f_loc = np.zeros((self.sample_num, 5))

        # formulate self.sample_num samples, and write them into 'self.f_wr' and 'self.loc'
        self.formulate_wr_loc()

        # update global matrix 'wr_container' and 'M_count'
        update_global_matrix(self.f_wr, self.f_loc)

        # Save standard input and output into files
        self.save_overall_txt()
        self.save_overall_hdf5()

    def first_parse_file(self, file_name):
        dom = xml.dom.minidom.parse(file_name)
        root = dom.documentElement

        wr_list = root.getElementsByTagName('wr')
        loc_list = root.getElementsByTagName('loc')

        print("# wifi record:", wr_list.length)
        print("# loc record:", loc_list.length)

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

        # the start and the end time of loc
        self.t1 = next(iter(self.loc_dict))[-1]  # get the first key and then the extract the final 'time' element
        self.t2 = next(reversed(self.loc_dict))[-1]  # get the last key and then the extract the final 'time' element

        self.new_wr_dict = collections.OrderedDict()
        for key, value in self.wr_dict.items():
            if (key > self.t1) & (key < self.t2):
                self.new_wr_dict[key] = value

        # the start and the end time of wr
        self.t11 = next(iter(self.new_wr_dict))  # get the first key
        self.t22 = next(reversed(self.new_wr_dict))  # get the last key

    def formulate_wr_loc(self):
        loc_time_list = list(self.loc_dict.keys())
        loc_latlng_list = list(self.loc_dict.values())
        cur_total = 0
        for i in range(len(self.loc_dict) - 1):
            # 算该线段的方向 calculate the direction
            delta_latlng = list(map(lambda x: x[0] - x[1], zip(loc_latlng_list[i + 1], loc_latlng_list[i])))

            # 选出区间内的wr。select the wr collected inside this range
            t1 = loc_time_list[i][1]
            t2 = loc_time_list[i + 1][1]
            for k, v in self.new_wr_dict.items():
                if k > t1 and k < t2:  # 如果该wr在loc1和loc2之间
                    percentage = (k - t1) / (t2 - t1)
                    delta_me = [xx * percentage for xx in delta_latlng]
                    interpolated_position = list(map(lambda x: x[0] + x[1], zip(loc_latlng_list[i], delta_me)))
                    lat = interpolated_position[0]
                    lng = interpolated_position[1]

                    self.f_wr[cur_total, :] = SensorFile.formalize_wr(v)
                    self.f_loc[cur_total, :] = SensorFile.latlng_to_grid(lat, lng)
                    cur_total = cur_total + 1
                elif k > t2:
                    break

# '''
#     # 每处理一个background文件调用一次该函数
#     def threshold_and_filter(self):
#
#         for k, v in self.loc_dict.items():
#             # for each recorded location
#             # 为每个loc筛选出符合时间条件的传感器数据
#             time_start = k[1] - 2000
#             time_end = k[1]
#
#             # -------------Select 2 sec previous sensor data for acc/mag---------------
#             acc_threshold = [(tt, vv) for tt, vv in self.acc_dict.items() if time_start < tt[0] < time_end]
#             mag_threshold = [(tt, vv) for tt, vv in self.mag_dict.items() if time_start < tt[0] < time_end]
#             (acc_reduced, mag_reduced) = reduce_frequency_average(time_start, time_end, acc_threshold, mag_threshold)
#
#             # --------------Select closest wifi record for each location-----------------
#             wr_reduced = self.select_closest_wr(time_end)
#
#             # ------------------------------------------------------------------------------------------------
#             # following is save standardised input/output data into object variables self.inputs/self.outputs
#             # k[0] indicates the index of this sample in this background file
#
#             # add acc(0:300) into f_inputs
#             for acc, i in zip(acc_reduced, range(len(acc_reduced))):
#                 self.f_inputs[k[0], 3 * i:3 * (i + 1)] = acc
#
#             # add mag(300:600) into f_inputs
#             for mag, i in zip(mag_reduced, range(len(mag_reduced))):
#                 self.f_inputs[k[0], 300 + 3 * i: 300 + 3 * (i + 1)] = mag
#
#             # add wr(600:702) into f_inputs
#             self.f_inputs[k[0], 600:] = np.array(wr_reduced).reshape(1, len(SensorFile.world_ap_dict.keys()))
#
#             # add (lat, lng, x, y, index) into f_outputs
#             self.f_outputs[k[0]] = np.array(SensorFile.latlng_to_grid(v[0], v[1]))
#
#     def select_closest_wr(self, loc_time):
#         try:
#             wr_selected = self.wr_dict[loc_time]
#         except KeyError:
#             # pre_dis = time_end - next(iter(self.wr_dict))
#             for wr_t in self.wr_dict.keys():  # self.wr_dict is an ordered dict
#                 cur_dis = loc_time - wr_t
#                 # 仍然没到time_end,扫描略过之前的所有wr,记住time_end的前一个wr
#                 if cur_dis > 0:
#                     pre_dis = cur_dis
#                     wr_selected = self.wr_dict[wr_t]
#                     continue
#                 # 读到了time_end后的那个wr，比较前面一个wr和后面一个wr，哪个更近
#                 else:
#                     # before "time_end" is the closest
#                     if pre_dis + cur_dis < 0:
#                         tt = loc_time - pre_dis
#                     # after "time_end" is the closest
#                     else:
#                         tt = loc_time - cur_dis
#                     wr_selected = self.wr_dict[tt]
#                     break
#         # formalise the selected wifi record
#         return SensorFile.formalize_wr(wr_selected)
# '''

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

    # write the overall standard input and output into a single "out_in_overall.txt" file
    def save_overall_txt(self):
        txt_filename = "./background_results/interpolation_loc_wr.txt"
        write_text = np.hstack((self.f_loc, self.f_wr))
        with open(txt_filename, "ab") as f:
            np.savetxt(f, write_text, delimiter=",", newline='\n')

    # write the overall standard input and output into a single "out_in_overall.h5" file
    def save_overall_hdf5(self):
        h5_filename = "./background_results/interpolation_loc_wr.h5"
        h5_file = h5py.File(h5_filename, mode='a')
        write_content = np.hstack((self.f_loc, self.f_wr))
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


# calculate the median wr(every ap) value for each transition
def cal_median_for_each_transition():
    median_container = {}

    # 对于wr_container中的每一个转场记录，也就是M_count中每一个不为0的元素
    for trans, small_dicts_list in wr_container.items():
        element = np.zeros((NUM_COLUMNS))
        # 填充element的每一个元素
        for ap in range(NUM_COLUMNS):
            # 对于转场trans所对应的第ap个AP
            small_dict = small_dicts_list[ap]
            # 找出small_dict中value最大的的key,即找出出现最多次的strength
            element[ap] = sorted(small_dict, key=lambda x: small_dict[x])[-1]
        median_container[trans] = element

    return median_container

# return 'M_probability' (a 237*237 matrix)
def generate_probability_matrix(m_count):
    matrix = np.zeros(np.shape(m_count))
    num_grids = np.shape(matrix)[0]
    # start from i
    for i in range(num_grids):
        element = np.zeros((num_grids))
        summation = sum(m_count[i, :])
        if summation != 0:
            # end in j
            for j in range(num_grids):
                element[j] = m_count[i, j] / summation
        matrix[i, :] = element
    return matrix



# saved 'trans_to_median_wr_dict' into "./background_results/M_median_wr.pkl"
# saved 'M_probability' into './background_results/M_prob.h5'
def save_matrix():
    # save 'M_probability' into file
    h5_filename = "./background_results/M_probability"
    h5_file = h5py.File(h5_filename, mode='w')
    write_content = M_probability
    h5_file.create_dataset('m_probability', data=write_content)
    h5_file.close()

    # save 'trans_to_median_wr_dict' into file
    pickle_file = open("./background_results/M_median_wr.pkl", mode='wb+')
    write_content = trans_to_median_wr_dict
    pkl.dump(write_content, pickle_file)
    pickle_file.close()


file1 = "./background_results/interpolation_loc_wr.h5"
file2 = "./background_results/interpolation_loc_wr.txt"
if os.path.isfile(file1):
    os.remove(file1)
if os.path.isfile(file2):
    os.remove(file2)

iterate("/Users/chenxingji/PycharmProjects/InLoc_preprocessing/background")
trans_to_median_wr_dict = cal_median_for_each_transition()
M_probability = generate_probability_matrix(M_count)
save_matrix()

print("\nFinished generating 2 global matrix...\n1: trans_to_median_wr_dict, 2: M_probability")
print("and save into separate file... \n1:'./background_results/M_median_wr.pkl', 2: './background_results/M_prob.h5'")
