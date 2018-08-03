import numpy as np
import matplotlib.pyplot as plt
import math

import statsmodels.api as sm
north_west = (55.945139, -3.18781)  # A
south_east = (55.944600, -3.186537)  # B
X_GRID_NUM = 40
Y_GRID_NUM = 30


def get_distance(lnglat1, lnglat2):
    '''
    get the distance in meters of two location
    :param lnglat1:
    :param lnglat2:
    :return: distance in meters
    '''
    rr = 6381 * 1000

    lng1 = lnglat1[0]
    lat1 = lnglat1[1]

    lng2 = lnglat2[0]
    lat2 = lnglat2[1]

    lng_distance = math.radians(lng2 - lng1)
    lat_distance = math.radians(lat2 - lat1)

    a = pow(math.sin(lat_distance / 2), 2) \
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) \
        * pow(math.sin(lng_distance / 2), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = rr * c
    return distance


def transfer_error_in_meters(data, flag):
    # flag is True, when neural net outputs 1 value(classification)
    if flag:
        index_xy = np.zeros((np.shape(data)[0], np.shape(data)[1] * 2))
        lnglat = np.zeros((np.shape(data)[0], np.shape(data)[1] * 2))

        # target
        index_xy[:, 0] = data[:, 0] // X_GRID_NUM
        index_xy[:, 1] = data[:, 0] % X_GRID_NUM
        # prediction
        index_xy[:, 2] = data[:, 1] // X_GRID_NUM
        index_xy[:, 3] = data[:, 1] % X_GRID_NUM

        delta_lng = (south_east[1] - north_west[1]) / X_GRID_NUM
        delta_lat = (north_west[0] - south_east[0]) / Y_GRID_NUM

        # target lng-x
        lnglat[:, 0] = north_west[1] + index_xy[:, 0] * delta_lng
        # target lat-y
        lnglat[:, 1] = north_west[0] - index_xy[:, 1] * delta_lat

        # prediction lng-x
        lnglat[:, 2] = north_west[1] + index_xy[:, 2] * delta_lng
        # prediction lat-y
        lnglat[:, 3] = north_west[0] - index_xy[:, 3] * delta_lat

        errors = np.zeros((np.shape(lnglat)[0], 1))

        for item, i in zip(lnglat, range(np.shape(errors)[0])):
            errors[i, 0] = get_distance([item[0], item[1]], [item[2], item[3]])

    # flag is False, when neural net outputs 2 value(regression)
    # the passes in "data" has 4 cols,
    else:
        lnglat = np.zeros(np.shape(data))
        # lng
        lnglat[:, 0] = north_west[1] + ((data[:, 0] + 1) * (south_east[1] - north_west[1]) / 2)
        lnglat[:, 2] = north_west[1] + ((data[:, 2] + 1) * (south_east[1] - north_west[1]) / 2)

        # lat
        lnglat[:, 1] = north_west[0] - ((data[:, 1] + 1) * (north_west[0] - south_east[0]) / 2)
        lnglat[:, 3] = north_west[0] - ((data[:, 3] + 1) * (north_west[0] - south_east[0]) / 2)

        errors = np.zeros((np.shape(lnglat)[0], 1))

        for item, i in zip(lnglat, range(np.shape(errors)[0])):
            errors[i, 0] = get_distance([item[0], item[1]], [item[2], item[3]])

    return errors


# \\wrong version, do not use
# def plot_cdf(mark):

# read neural(target & prediction) from "test_output_{}.txt",
# and write error in meters into file "e_{}.txt"
def save_error_in_meters(mark, which_comparison, fn_suffix):
    '''

    :param mark: 1 - classification, 2 - regression
    :param fn_suffix: identify which model has been used(classification/regression, what is the hidden layer's structure)
    :return: no return, but write error in meters in file "./interim_output/e_{}.txt"
    '''
    fn = './comparison{}/test_output_{}.txt'.format(which_comparison,fn_suffix)
    # experiment = ["classification", "regression"]
    target_and_neural_out = np.loadtxt(fn, delimiter=',')

    # classification
    if mark == 1:
        error_in_meters = transfer_error_in_meters(target_and_neural_out, flag=True)
        with open("./comparison{}/e_{}.txt".format(which_comparison, fn_suffix), "w") as f:
            np.savetxt(f, error_in_meters, delimiter=",", newline='\n')

    # regression
    else:
        error_in_meters = transfer_error_in_meters(target_and_neural_out, flag=False)
        with open("./comparison{}/e_{}.txt".format(which_comparison, fn_suffix), "w") as f:
            np.savetxt(f, error_in_meters, delimiter=",", newline='\n')

    # # Choose how many bins you want here
    # num_bins = 20
    #
    # # Use the histogram function to bin the data
    # counts, bin_edges = np.histogram(error_in_meters, bins=num_bins, normed=True)
    #
    # # Find the cdf
    # cdf = np.cumsum(counts, )
    #
    # # Finally plot the cdf
    # fig = plt.figure()
    # plt.plot(bin_edges[1:], cdf)
    # plt.xlabel("Error in meters")
    # plt.ylabel("cdf")
    # plt.title("Neural net output {} value({})".format(mark, experiment[mark-1]))
    # plt.show()
    # fig.savefig("cdf_{}.png".format(mark))


# def plot_cdf_2(fn1='./interim_output/test_output_1.txt', fn2='./interim_output/test_output_2.txt'):
#
#     output1_in_neural = np.loadtxt(fn1, delimiter=',')
#     output2_in_neural = np.loadtxt(fn2, delimiter=',')
#
#     error_in_meters1 = transfer_error_in_meters(output1_in_neural, flag=True)
#     error_in_meters2 = transfer_error_in_meters(output2_in_neural, flag=False)
#
#     # Choose how many bins you want here
#     num_bins = 20
#
#     # Use the histogram function to bin the data
#     counts1, bin_edges1 = np.histogram(error_in_meters1, bins=num_bins, normed=True)
#     counts2, bin_edges2 = np.histogram(error_in_meters2, bins=num_bins, normed=True)
#
#     # Find the cdf
#     cdf1 = np.cumsum(counts1, )
#     cdf2 = np.cumsum(counts2, )
#
#     # Finally plot the cdf
#     fig = plt.figure()
#     plt.plot(bin_edges1[1:], cdf1)
#     plt.xlabel("Error in meters")
#     plt.ylabel("cdf")
#     plt.title("Neural net output 1 value(classification)")
#     plt.show()
#     fig.savefig("cdf_1.png")
#
#     fig = plt.figure()
#     plt.plot(bin_edges2[1:], cdf2)
#     plt.xlabel("Error in meters")
#     plt.ylabel("cdf")
#     plt.title("Neural net output 2 value(regression)")
#     plt.show()
#     fig.savefig("cdf_2.png")


def plot_train_val(model_history, is_classification, suffix):

    fig = plt.figure(figsize=(6, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(model_history['loss'], label="train_loss")
    ax1.plot(model_history['val_loss'], label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(model_history['acc'], label="train_acc")
    ax2.plot(model_history['val_acc'], label="val_acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    if is_classification:
        ax1.set_title("classification model{}".format(suffix))
        plt.show()
        fig.savefig("./graph_output/Acc_Loss_curve_cla_{}.png".format(suffix))
    else:
        ax1.set_title("regression model{}".format(suffix))
        plt.show()
        fig.savefig("./graph_output/Acc_Loss_curve_reg_{}.png".format(suffix))


# correct implementation version
def cdf_plot(data, name, number):
    ecdf = sm.distributions.ECDF(data)

    # x = np.linspace(min(data), max(data), number)
    lower_bound = ecdf.x[int(620*0.1)]
    upper_bound = ecdf.x[int(620*0.9)]
    x = np.linspace(lower_bound, upper_bound, number)
    y = ecdf(x)

    print("x ~ [{}, {}]".format(round(ecdf.x[0], 3), round(ecdf.x[620], 3)))
    print("mid_x={}\t\t{}\n".format(round(ecdf.x[310], 3), name))


    # plt.step(x, y, label=name)
    plt.plot(x, y, label=name)


# 该函数将网络输出结果和目标对比的文件"interim_output/test_output_{}.txt"转为米计误差，并写入"./error_output/e_{}.txt"
def main(which_kind_of_model):
    if which_kind_of_model is "m1":
        added_label = ""
    elif which_kind_of_model is "m2":
        added_label = "+dropout"
    elif which_kind_of_model is "m3":
        added_label = "+autoencoder"
    elif which_kind_of_model is "m4":
        added_label = "+dropout+auto"

    fn_suffix_list = ["1", "2", "3", "4"]

    fig = plt.figure()
    models = ["C{}(64,32,16)".format(added_label), "C{}(200,200,200)".format(added_label)]
    for suffix, model_name in zip(fn_suffix_list[0:2], models):
        data = np.loadtxt("./errors/errors_{}_{}.txt".format(which_kind_of_model, suffix))
        cdf_plot(data, model_name, 100)
    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
    plt.show()
    fig.savefig("cdf_{}_1.png".format(which_kind_of_model))
    print("\n")

    fig = plt.figure()
    models = ["R{}(64,32,16)".format(added_label), "R{}(200,200,200)".format(added_label)]
    for suffix, model_name in zip(fn_suffix_list[2:], models):
        data = np.loadtxt("./errors/errors_{}_{}.txt".format(which_kind_of_model, suffix))
        cdf_plot(data, model_name, 100)
    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
    plt.show()
    fig.savefig("cdf_{}_2.png".format(which_kind_of_model))
    print("\n")

    fig = plt.figure()
    models = ["C{}(64,32,16)".format(added_label), "C{}(200,200,200)".format(added_label),
              "R{}(64,32,16)".format(added_label), "R{}(200,200,200)".format(added_label)]
    for suffix, model_name in zip(fn_suffix_list, models):
        data = np.loadtxt("./errors/errors_{}_{}.txt".format(which_kind_of_model, suffix))
        cdf_plot(data, model_name, 100)
    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
    plt.show()
    fig.savefig("cdf_{}_0.png".format(which_kind_of_model))
    print("\n")



# comparison1: "simple" vs "simple+dropout"
def main1():
    fn_suffix_list = ["m1_1", "m1_1_1", "m1_2", "m1_2_1",
                      "m2_1", "m2_1_1", "m2_2", "m2_2_1"]

    # classification(64, 32, 16)
    save_error_in_meters(mark=1, which_comparison=1, fn_suffix=fn_suffix_list[0])
    save_error_in_meters(mark=1, which_comparison=1, fn_suffix=fn_suffix_list[4])

    # classification(200,200,200)
    save_error_in_meters(mark=1, which_comparison=1, fn_suffix=fn_suffix_list[1])
    save_error_in_meters(mark=1, which_comparison=1, fn_suffix=fn_suffix_list[5])

    # regression(64, 32, 16)
    save_error_in_meters(mark=2, which_comparison=1, fn_suffix=fn_suffix_list[2])
    save_error_in_meters(mark=2, which_comparison=1, fn_suffix=fn_suffix_list[6])

    # regression(200,200,200)
    save_error_in_meters(mark=2, which_comparison=1, fn_suffix=fn_suffix_list[3])
    save_error_in_meters(mark=2, which_comparison=1, fn_suffix=fn_suffix_list[7])

    fig = plt.figure()
    models = ["C(64,32,16)", "C(200,200,200)", "R(64,32,16)", "R(200,200,200)",
              "C+dropout(64,32,16)", "C+dropout(200,200,200)", "R+dropout(64,32,16)", "R+dropout(200,200,200)"]
    for suffix, model_name in zip(fn_suffix_list, models):
        data = np.loadtxt("./comparison1/e_{}.txt".format(suffix))
        cdf_plot(data, model_name, 100)

    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
    plt.show()
    fig.savefig("CDF1.png")
    print("\n")



# comparison1: "simple" vs "simple+autoencoder"
def main2():
    fn_suffix_list = ["m1_1", "m1_1_1", "m1_2", "m1_2_1",
                      "m2_1", "m2_1_1", "m2_2", "m2_2_1"]

    # classification(64, 32, 16)
    save_error_in_meters(mark=1, which_comparison=2, fn_suffix=fn_suffix_list[0])
    save_error_in_meters(mark=1, which_comparison=2, fn_suffix=fn_suffix_list[4])

    # classification(200,200,200)
    save_error_in_meters(mark=1, which_comparison=2, fn_suffix=fn_suffix_list[1])
    save_error_in_meters(mark=1, which_comparison=2, fn_suffix=fn_suffix_list[5])

    # regression(64, 32, 16)
    save_error_in_meters(mark=2, which_comparison=2, fn_suffix=fn_suffix_list[2])
    save_error_in_meters(mark=2, which_comparison=2, fn_suffix=fn_suffix_list[6])

    # regression(200,200,200)
    save_error_in_meters(mark=2, which_comparison=2, fn_suffix=fn_suffix_list[3])
    save_error_in_meters(mark=2, which_comparison=2, fn_suffix=fn_suffix_list[7])

    fig = plt.figure()
    models = ["C(64,32,16)", "C(200,200,200)", "R(64,32,16)", "R(200,200,200)",
              "C+autoencoder(64,32,16)", "C+autoencoder(200,200,200)", "R+autoencoder(64,32,16)", "R+autoencoder(200,200,200)"]
    for suffix, model_name in zip(fn_suffix_list, models):
        data = np.loadtxt("./comparison2/e_{}.txt".format(suffix))
        cdf_plot(data, model_name, 100)

    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
    plt.show()
    fig.savefig("CDF2.png")
    print("\n")



# comparison3: "auto" vs "auto+dropout"
def main3():
    fn_suffix_list = ["m1_1", "m1_1_1", "m1_2", "m1_2_1",
                      "m2_1", "m2_1_1", "m2_2", "m2_2_1"]

    # classification(64, 32, 16)
    save_error_in_meters(mark=1, which_comparison=3, fn_suffix=fn_suffix_list[0])
    save_error_in_meters(mark=1, which_comparison=3, fn_suffix=fn_suffix_list[4])

    # classification(200,200,200)
    save_error_in_meters(mark=1, which_comparison=3, fn_suffix=fn_suffix_list[1])
    save_error_in_meters(mark=1, which_comparison=3, fn_suffix=fn_suffix_list[5])

    # regression(64, 32, 16)
    save_error_in_meters(mark=2, which_comparison=3, fn_suffix=fn_suffix_list[2])
    save_error_in_meters(mark=2, which_comparison=3, fn_suffix=fn_suffix_list[6])

    # regression(200,200,200)
    save_error_in_meters(mark=2, which_comparison=3, fn_suffix=fn_suffix_list[3])
    save_error_in_meters(mark=2, which_comparison=3, fn_suffix=fn_suffix_list[7])

    fig = plt.figure()
    models = ["C+auto(64,32,16)", "C+auto(200,200,200)", "R+auto(64,32,16)", "R+auto(200,200,200)",
              "C+auto+dropout(64,32,16)", "C+auto+dropout(200,200,200)", "R+auto+dropout(64,32,16)", "R+auto+dropout(200,200,200)"]
    for suffix, model_name in zip(fn_suffix_list, models):
        data = np.loadtxt("./comparison3/e_{}.txt".format(suffix))
        cdf_plot(data, model_name, 100)

    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.3), borderaxespad=0.)
    plt.show()
    fig.savefig("CDF3.png")
    print("\n")


if __name__ == '__main__':
    # main1()
    # main2()
    # main3()
    main("m1")
    main("m2")
    main("m3")
    main("m4")
