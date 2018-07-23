import numpy as np

np.random.seed(100)
import tensorflow as tf

tf.set_random_seed(100)


import h5py
from preprocessing import Masking
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from collections import OrderedDict
import Plotting
import sdae


north_west = (55.945139, -3.18781)  # A
south_east = (55.944600, -3.186537)  # B
X_GRID_NUM = 40
Y_GRID_NUM = 30

# -------------- (1.Classification) Some function used in main1 -----------------
def normalize_inputs(inputs):
    # normalise wifi record strength
    wr_inputs = inputs[:, 600:]
    zero_index = np.where(wr_inputs == 0)
    wr_inputs[zero_index] = -100

    max = np.max(wr_inputs)
    min = np.min(wr_inputs)

    wr_inputs = (wr_inputs - min) / (max - min)

    return wr_inputs


def int_to_categorical(mask, y):
    y = np.array(y, dtype='int')
    y = y.ravel()

    # get distinct grid index
    # exist_grid_list = sorted(set(y))
    exist_grid_list = []
    y_list, x_list = np.where(mask == 1)
    for i, j in zip(x_list, y_list):
        index = j * X_GRID_NUM + i
        exist_grid_list.append(index)

    for yy in y:
        if yy not in exist_grid_list:
            exist_grid_list.append(yy)
            print("add extra...{}".format(yy))
    exist_grid_list = sorted(exist_grid_list)

    exist_grid_label_dict = {}
    for ind, label in zip(exist_grid_list, range(len(exist_grid_list))):
        exist_grid_label_dict[ind] = label

    num_classes = len(exist_grid_list)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)

    for yy, i in zip(y, range(n)):
        categorical[i, exist_grid_label_dict[yy]] = 1

    return exist_grid_list, categorical


def data_shuffle_split1(name):
    dataset = h5py.File(name, "r")

    # 将h5文件中所有的dataset(即所有app输出的file)合并成一个X_Y
    for name, i in zip(dataset.keys(), range(len(dataset.keys()))):
        print(np.shape(dataset[name]))
        if i:
            X_Y = np.vstack((X_Y, dataset[name]))
        else:
            X_Y = dataset[name]
        print("--------")
    dataset.close()
    np.random.shuffle(X_Y)
    SAMPLE_NUM = np.shape(X_Y)[0]
    INPUT_DIM = np.shape(X_Y)[1] - 5 - 2 * 300

    all_X = X_Y[:, 5:]
    X = normalize_inputs(all_X)

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
    grid_list, Y = int_to_categorical(array, X_Y[:, 4])  # using reserved grids index to align with the standardized
    # output and Simultaneously convert numeric class labels to vectors Y
    OUTPUT_DIM = len(grid_list)

    chunk_size = int(0.2 * SAMPLE_NUM)
    Y_test = Y[0:chunk_size, :]
    X_test = X[0:chunk_size, :]

    Y_train = Y[chunk_size:, :]
    X_train = X[chunk_size:, :]

    return X_train, Y_train, X_test, Y_test, grid_list, SAMPLE_NUM, INPUT_DIM, OUTPUT_DIM


# --------------------------------------------------------------------------------


# --------------- (2.Regression) Some function used in main2 ---------------------
# (may not be useful now) For xy index classification
def generate_concat_xy(coordinates):
    coor_x = np.zeros((np.shape(coordinates)[0], X_GRID_NUM), dtype='int')
    coor_y = np.zeros((np.shape(coordinates)[0], Y_GRID_NUM), dtype='int')

    for i in range(len(coordinates)):
        coor_x[i, int(coordinates[i, 0])] = 1
        coor_y[i, int(coordinates[i, 1])] = 1

    coor_xy = np.hstack((coor_x, coor_y))

    return coor_xy


# the parameter taking in this function has 2 cols, the first one represents latitude, second one represent
# longitude, but the return of this function has been reversed, which means the first column represent x,
# and the second represent y
def normalize_outputs(outputs):
    # lat-y
    # max0 = north_west[0]
    # min0 = south_east[0]
    # mm0 = (max0 + min0) / 2
    # outputs[:, 0] = 2 * (outputs[:, 0] - mm0) / (max0 - min0)
    max0 = north_west[0]
    min0 = south_east[0]
    outputs[:, 0] = 2 * (outputs[:, 0] - min0) / (max0 - min0) - 1

    # lng-x
    max1 = south_east[1]
    min1 = north_west[1]
    outputs[:, 1] = 2 * (outputs[:, 1] - min1) / (max1 - min1) - 1

    # now the outputs[lat, lng], that is [y, x]
    # we would like to reverse the order of the 2 columns, and become [x,y] as follow:
    outputs[:, [0, 1]] = outputs[:, [1, 0]]

    return outputs


def data_shuffle_split2(name):
    dataset = h5py.File(name, "r")

    # write structured data from "dataset"(h5py file) into "X_Y"(numpy array)
    for name, i in zip(dataset.keys(), range(len(dataset.keys()))):
        print(np.shape(dataset[name]))
        if i:
            X_Y = np.vstack((X_Y, dataset[name]))
        else:
            X_Y = dataset[name]
        print("--------")
    dataset.close()

    np.random.shuffle(X_Y)
    SAMPLE_NUM = np.shape(X_Y)[0]
    INPUT_DIM = np.shape(X_Y)[1] - 5 - 2 * 300
    OUTPUT_DIM = 2

    # get formatted nueral network inputs(standardised wifi signal)- 92 values
    X = X_Y[:, 5:]
    X = normalize_inputs(X)  # get normalized wifi inputs(The parameters passed in by this function are including the
    #  acc,mag and wifi signal, while only return the standardised wifi signal.)

    # get formatted and normalised target outputs(x,y)- 2 values
    Y = X_Y[:, :2]
    Y = normalize_outputs(Y)

    chunk_size = int(0.2 * SAMPLE_NUM)
    Y_test = Y[0:chunk_size, :]
    X_test = X[0:chunk_size, :]

    Y_train = Y[chunk_size:, :]
    X_train = X[chunk_size:, :]

    return X_train, Y_train, X_test, Y_test, SAMPLE_NUM, INPUT_DIM, OUTPUT_DIM


# wrong version of visualization function, use the next one
'''def visualization(grid_dict, Y_test, Y_pre, suffix):
    fig = plt.figure()
    plt.xlabel("x-longitude")
    plt.ylabel("y-latitude")
    # output 1 (classification)
    if grid_dict is not None:
        for target, pred in zip(Y_test, Y_pre):
            target = target.tolist()
            pred = pred.tolist()

            tar_ind = target.index(max(target))
            pre_ind = pred.index(max(pred))


            # error line
            plt.plot([grid_dict[pre_ind] % X_GRID_NUM, grid_dict[tar_ind] % X_GRID_NUM],
                     [grid_dict[pre_ind] // X_GRID_NUM, grid_dict[tar_ind] // X_GRID_NUM], label='error line', color='r', linewidth=0.5)
            # prediction point
            plt.scatter(grid_dict[pre_ind] % X_GRID_NUM, grid_dict[pre_ind] // X_GRID_NUM, label='prediction', color='b', marker='.')
            # target point
            plt.scatter(grid_dict[tar_ind] % X_GRID_NUM, grid_dict[tar_ind] // X_GRID_NUM, label='target', color='c', marker='.')

        # plt.title("Errors of classification(64, 32, 16)")
        plt.title("Errors of classification{}".format(suffix))

        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = OrderedDict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys())

        plt.show()
        # fig.savefig('./graph_output/errors_visualization_1.png')    # classification [64,32,16]
        fig.savefig('./graph_output/errors_visualization_1_1.png')    # calssification [200,200,200]

    # output 2 values (regression)
    else:
        for target, pred in zip(Y_test, Y_pre):
            # tt = "{:.3f}, {:.3f}".format(float(target[0]), float(target[1]))
            # pp = "{:.3f}, {:.3f}".format(float(pred[0]), float(pred[1]))
            # print("target:({})  predict:({})".format(tt, pp))

            # tt = "({}, {})".format(int(target[0] * 40), int(target[1] * 30))
            # pp = "({}, {})".format(int(pred[0] * 40), int(pred[1] * 30))
            # print("target:{}  predict:{}\n".format(tt, pp))

            plt.plot([int(pred[0] * 40), int(target[0] * 40)], [int(pred[1] * 30), int(target[1] * 30)], color='r',
                     linewidth=0.5, label='error line')
            plt.scatter(int(pred[0] * 40), int(pred[1] * 30), label='prediction', color='b', marker='.')
            plt.scatter(int(target[0] * 40), int(target[1] * 30), label='target', color='c', marker='.')

        plt.title("Errors of regression")

        # handles, labels = plt.gca().get_legend_handles_labels()
        handles, labels = plt.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.show()
        # fig.savefig('./graph_output/errors_visualization_2.png')    # regression [64,32,16]
        fig.savefig('./graph_output/errors_visualization_2_1.png')    # regression [200,200,200]

'''

# error line plot into file "./graph_output/errors_visualization_{}.png"
def visualization(grid_dict, Y_test, Y_pre, baseline, suffix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlabel("x-longitude")
    ax.set_ylabel("y-latitude")

    # xmajorLocator = MultipleLocator(5)  # 将x主刻度标签设置为10的倍数
    # xmajorFormatter = FormatStrFormatter('%1.f')  # 设置x轴标签文本的格式
    # xminorLocator = MultipleLocator(1)  # 将x轴次刻度标签设置为2的倍数
    # ymajorLocator = MultipleLocator(5)  # 将x主刻度标签设置为10的倍数
    # ymajorFormatter = FormatStrFormatter('%1.f')  # 设置x轴标签文本的格式
    # yminorLocator = MultipleLocator(1)  # 将x轴次刻度标签设置为2的倍数
    #
    # # setting x,y axis major
    # ax.xaxis.set_major_locator(xmajorLocator)
    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # ax.yaxis.set_major_locator(ymajorLocator)
    # ax.yaxis.set_major_formatter(ymajorFormatter)
    #
    # # setting x,y axis minor (do not have label text)
    # ax.xaxis.set_minor_locator(xminorLocator)
    # ax.yaxis.set_minor_locator(yminorLocator)
    #
    # # 均使用次坐标轴画网格
    # ax.xaxis.grid(True, which='minor')
    # ax.yaxis.grid(True, which='minor')


    # output 1 (classification)
    if grid_dict is not None:

        # 设置x,y主坐标轴
        my_x_ticks = np.arange(0, 40, 5)
        my_y_ticks = np.arange(0, 30, 5)
        ax.set_xticks(my_x_ticks, minor=False)
        ax.set_yticks(my_y_ticks, minor=False)
        # 设置x,y次坐标轴
        my_x_ticks = np.arange(0, 40, 1)
        my_y_ticks = np.arange(0, 30, 1)
        ax.set_xticks(my_x_ticks, minor=True)
        ax.set_yticks(my_y_ticks, minor=True)

        ax.set_xlim((0, 40))
        ax.set_ylim((0, 30))

        # 设置x，y值域
        ax.set_xlim(left=0, right=40)
        ax.set_ylim(bottom=30, top=0)  # 此处将原点设置为左上角
        ax.xaxis.tick_top()  # 将x坐标标记移到上方

        # 均使用次坐标轴画网格
        ax.xaxis.grid(True, which='minor')
        ax.yaxis.grid(True, which='minor')

        for target, pred, i in zip(Y_test, Y_pre, range(np.shape(Y_test)[0])):
            target = target.tolist()
            pred = pred.tolist()

            tar_ind = target.index(max(target))
            pre_ind = pred.index(max(pred))

            # error line
            ax.plot([grid_dict[pre_ind] % X_GRID_NUM, grid_dict[tar_ind] % X_GRID_NUM],
                     [grid_dict[pre_ind] // X_GRID_NUM, grid_dict[tar_ind] // X_GRID_NUM], label='error line' if i == 0 else "",
                    color='r', linewidth=0.5)
            # prediction point
            ax.scatter(grid_dict[pre_ind] % X_GRID_NUM, grid_dict[pre_ind] // X_GRID_NUM, label='prediction' if i == 0 else "",
                       color='b', marker='.')
            # target point
            ax.scatter(grid_dict[tar_ind] % X_GRID_NUM, grid_dict[tar_ind] // X_GRID_NUM, label='target' if i == 0 else "",
                       color='c', marker='.')

        ax.legend()
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = OrderedDict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys())

        plt.title("Errors of classification{}".format(suffix), y=1.08)
        plt.show()

        # save fig
        if baseline:
            fig.savefig('./graph_output/errors_visualization_1_1.png')  # calssification [200,200,200]
        else:
            fig.savefig('./graph_output/errors_visualization_1.png')    # classification [64,32,16]


    # output 2 values (regression)
    else:
        # 设置x,y主坐标轴
        my_x_ticks = np.arange(-40, 40, 10)
        my_y_ticks = np.arange(-30, 30, 10)
        ax.set_xticks(my_x_ticks, minor=False)
        ax.set_yticks(my_y_ticks, minor=False)

        # 设置x,y次坐标轴
        my_x_ticks = np.arange(-40, 40, 2)
        my_y_ticks = np.arange(-30, 30, 2)
        ax.set_xticks(my_x_ticks, minor=True)
        ax.set_yticks(my_y_ticks, minor=True)

        ax.set_xlim((-40, 40))
        ax.set_ylim((-30, 30))


        for target, pred, i in zip(Y_test, Y_pre, range(np.shape(Y_test)[0])):
            # tt = "{:.3f}, {:.3f}".format(float(target[0]), float(target[1]))
            # pp = "{:.3f}, {:.3f}".format(float(pred[0]), float(pred[1]))
            # print("target:({})  predict:({})".format(tt, pp))

            # tt = "({}, {})".format(int(target[0] * 40), int(target[1] * 30))
            # pp = "({}, {})".format(int(pred[0] * 40), int(pred[1] * 30))
            # print("target:{}  predict:{}\n".format(tt, pp))

            ax.plot([int(pred[0] * 40), int(target[0] * 40)], [int(pred[1] * 30), int(target[1] * 30)], color='r',
                     linewidth=0.5, label='error line' if i == 0 else "")
            ax.scatter(int(pred[0] * 40), int(pred[1] * 30),  color='b', marker='.', label='prediction' if i == 0 else "")
            ax.scatter(int(target[0] * 40), int(target[1] * 30), color='c', marker='.', label='target' if i == 0 else "")

        ax.set_title("Errors of regression{}".format(suffix))
        ax.legend()
        plt.show()

        # save fig
        if baseline:
            fig.savefig('./graph_output/errors_visualization_2_1.png')  # regression [200,200,200]
        else:
            fig.savefig('./graph_output/errors_visualization_2.png')    # regression [64,32,16]




# save neural output to file "./interim_output/test_output_{}.txt"
def save_results(grid_dict, Y_test, Y_pre, baseline):
    # output 1 values (classification)
    if grid_dict is not None:

        if baseline:
            txt_filename = "./interim_output/test_output_1_1.txt"   # classification [200,200,200]
        else:
            txt_filename = "./interim_output/test_output_1.txt"   # classification [64,32,16]

        tt = np.zeros((np.shape(Y_test)[0], 1))
        pp = np.zeros((np.shape(Y_pre)[0], 1))

        # the times of iterations = the number of samples in test set
        for target, pred, i in zip(Y_test, Y_pre, range(np.shape(Y_test)[0])):
            # target/pred here is a list of probabilities(length = num of classes)
            target = target.tolist()
            pred = pred.tolist()

            tar_ind = target.index(max(target))
            pre_ind = pred.index(max(pred))

            # find the original/overall(in Y_GRID_NUM*X_GRID_NUM) index
            tt[i] = grid_dict[tar_ind]
            pp[i] = grid_dict[pre_ind]

        write_text = np.hstack((tt, pp)).astype(np.int)

        # write the target and predicted overall grid index into "test_output_1.txt"
        with open(txt_filename, "wb") as f:
            np.savetxt(f, write_text, delimiter=",", newline='\n')

    # output 2 values (regression)
    else:
        if baseline:
            txt_filename = "./interim_output/test_output_2_1.txt"   # regression [200,200,200]
        else:
            txt_filename = "./interim_output/test_output_2.txt"   # regression [64,32,16]
        write_text = np.hstack((Y_test, Y_pre))

        # write the target output and predicted output into "test_output_2.txt"
        with open(txt_filename, "wb") as f:
            np.savetxt(f, write_text, delimiter=",", newline='\n')


# --------------------------------------------------------------------------------




# 1.Classification(outputs: grid_index [discrete]), which the hidden layer can be specified by passing a parameter to it
def main_classification_sdae(hidden_num,is_baseline):
    # Read data from file into memory
    f_name = "./background_results/out_in_overall(gridsize2).h5"
    x_train, y_train, x_test, y_test, grid_list, SAMPLE_NUM, INPUT_DIM, OUTPUT_DIM = data_shuffle_split1(f_name)

    pretrained_layer = sdae.train_SDAE(hidden_num)
    model = sdae.build_and_finetune_pretrained_classification(pretrained_layer=pretrained_layer, X_train=x_train,
                                                       Y_train=y_train, X_test=x_test, Y_test=y_test)

    # plot the training curve
    Plotting.plot_train_val(model_history=model.history.history, mark="cla_autoencoder{}".format(hidden_num))

    score = model.evaluate(x_test, y_test, batch_size=8)
    print(score)

    y_pre = model.predict(x_test, batch_size=8)
    # classification
    if 'grid_list' in vars():
        # generate the error line png
        visualization(grid_dict=grid_list, Y_test=y_test, Y_pre=y_pre, baseline=is_baseline, suffix=hidden_num)
        # save the tar&pre into "./interim_output/test_output_{}.txt"
        save_results(grid_dict=grid_list, Y_test=y_test, Y_pre=y_pre, baseline=is_baseline)  # baseline [200,200,200]
    # regression
    else:
        visualization(grid_dict=None, Y_test=y_test, Y_pre=y_pre, baseline=is_baseline, suffix=hidden_num)
        save_results(grid_dict=None, Y_test=y_test, Y_pre=y_pre, baseline=is_baseline)  # baseline [200,200,200]



# 2.Regression(outputs: grid_index [discrete]), which the hidden layer can be specified by passing a parameter to it
def main_regression_sdae(hidden_num,is_baseline):
    # Read data from file into memory
    f_name = "./background_results/out_in_overall(gridsize2).h5"
    x_train, y_train, x_test, y_test, SAMPLE_NUM, INPUT_DIM, OUTPUT_DIM = data_shuffle_split2(f_name)

    pretrained_layer = sdae.train_SDAE(hidden_num)
    model = sdae.build_and_finetune_pretrained_regression(pretrained_layer=pretrained_layer, X_train=x_train,
                                                       Y_train=y_train, X_test=x_test, Y_test=y_test)

    # plot the training curve
    Plotting.plot_train_val(model_history=model.history.history, mark="reg_autoencoder{}".format(hidden_num))

    score = model.evaluate(x_test, y_test, batch_size=8)
    print(score)

    y_pre = model.predict(x_test, batch_size=8)
    # classification
    if 'grid_list' in vars():
        visualization(grid_dict=grid_list, Y_test=y_test, Y_pre=y_pre, baseline=is_baseline, suffix=hidden_num)
        save_results(grid_dict=grid_list, Y_test=y_test, Y_pre=y_pre, baseline=is_baseline)  # baseline [200,200,200]
    # regression
    else:
        # generate the error line png
        visualization(grid_dict=None, Y_test=y_test, Y_pre=y_pre, baseline=is_baseline, suffix=hidden_num)
        # save the tar&pre into "./interim_output/test_output_{}.txt"
        save_results(grid_dict=None, Y_test=y_test, Y_pre=y_pre, baseline=is_baseline)  # baseline [200,200,200]




if __name__ == "__main__":

    # main_classification_sdae([64, 32, 16], is_baseline=False)
    # main_classification_sdae([200, 200, 200], is_baseline=True)
    # main_regression_sdae([64, 32, 16], is_baseline=False)
    main_regression_sdae([200, 200, 200], is_baseline=True)