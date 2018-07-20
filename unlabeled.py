import os
import xml.dom.minidom
import collections
import numpy as np
import re
import h5py



# ----------------------------------------------------------------------------------------

wifi_filename = "./preprocessing/wifi_id.txt"


def read_ap_to_dict(filename):
    ap_dict = collections.OrderedDict()
    with open(filename) as file:
        for line in file:
            elements = re.split(r'[\s]', line.strip())
            ap_dict[elements[0]] = (elements[1], elements[2])
    return ap_dict


WIFI_DICT = read_ap_to_dict(wifi_filename)


# ----------------------------------------------------------------------------------------

unlabeled_wifi_list = list()


def scan_wifi_and_write(file_name):
    dom = xml.dom.minidom.parse(file_name)
    root = dom.documentElement

    wr_list = root.getElementsByTagName('wr')
    for item, i in zip(wr_list, range(len(wr_list))):  # for each time step
        # i = i + 1
        # pre = item.getAttribute("t")
        # print(i, "->", item.getAttribute("t"), len(item.childNodes) // 2)
        wifi_record = []    # wifi_record is a list, which contains len(wifi_dict) elements
        for _ in range(len(WIFI_DICT.keys())):
            wifi_record.append(0)

        for record, j in zip(item.childNodes, range(len(item.childNodes))):  # for each AP
            if j % 2:
                ap_id = item.childNodes[j].getAttribute("b")
                ap_v = item.childNodes[j].getAttribute("s")
                if ap_id in WIFI_DICT.keys():
                    index = int(WIFI_DICT[ap_id][1])-1
                    wifi_record[index] = int(ap_v)
                else:
                    print("{} not in dict\n".format(ap_id))

        # 一条wifi_record生成,即一个unlabeled的样本生成，将其append到unlabeled_wifi_record
        unlabeled_wifi_list.append(wifi_record)
    # 每读完一个文件，输出当前unlabeled中样本的数目
    print("current sample size: {}\n".format(len(unlabeled_wifi_list)))


def traverse_all(path):
    '''
    traverse all background files and get all wifi records
    convert them into standard input and write those unlabeld wifi data into file
    :return:
    '''
    dirs = os.listdir(path)
    for dd in dirs:
        if dd != ".DS_Store":
            fi_d = os.path.join(path, dd)
            if os.path.isdir(fi_d):
                traverse_all(fi_d)
            else:
                print("processing... ", fi_d)
                scan_wifi_and_write(fi_d)


def normalize_wifi_inputs(wr_inputs):
    # normalise wifi record strength

    zero_index = np.where(wr_inputs == 0)
    wr_inputs[zero_index] = -100

    max = -40
    min = -100

    wr_inputs = (wr_inputs - min) / (max - min)

    return wr_inputs


traverse_all("./background")
data_shape = (len(unlabeled_wifi_list), len(WIFI_DICT.keys()))
unlabeled_wifi_records = np.reshape(unlabeled_wifi_list, data_shape)
unlabeled_wifi_array = normalize_wifi_inputs(unlabeled_wifi_records)

with open("./unlabeled/unlabeled_wifi.txt", "w") as f:
    np.savetxt(f, unlabeled_wifi_array, delimiter=",", newline='\n')

h5_filename = "./unlabeled/unlabeled_wifi.h5"
h5_file = h5py.File(h5_filename, mode='w')
h5_file.create_dataset("unlabeled neural inputs", data=unlabeled_wifi_array)
h5_file.close()

