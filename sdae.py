from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import sgd
from sklearn import preprocessing
import h5py
import numpy as np

'''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'
 # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
# 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
 # 只显示 Error
'''

INPUT_DIM = 102
NUM_CLASS = 237


# This function is used to train each layer of the SDAE, which means each layer in the neural nets should applied once
def train_layerwise_SDAE(X_train, input_dim, hidden_layer_size, noise_factor):
    '''

    :param X_train: The pure unlabeled input data
    :param input_size: the number of samples in the X_train
    :param hidden_layer_size: the number of units in this hidden layer
    :param noise_factor: the gaussian noise factor added to the inputs
    :return: Returns the trained hidden layer and the encoded output produced
    to apply in the next layer of the stack
    '''

    # Adding Gaussian noise to input data (for the denoising autoencoder)
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0,
                                                              scale=1.0,
                                                              size=X_train.shape)
    X_train_noisy = preprocessing.scale(X_train_noisy)

    # Training one layer of the SDAE: creating architecture
    input_layer = Input(shape=(input_dim,))
    hidden_layer = Dense(hidden_layer_size, activation="relu")(input_layer)
    output_layer = Dense(units=input_dim, activation="sigmoid")(hidden_layer)

    # Training one layer of the SDAE: fitting model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer="adadelta", loss="mean_squared_error")
    autoencoder.fit(X_train_noisy, X_train, epochs=5, batch_size=32, verbose=0)

    # Obtaining higher level representation of input by encoding it
    encoder = Model(inputs=input_layer, outputs=hidden_layer)
    encoded_output = encoder.predict(X_train, batch_size=32)
    encoded_output = preprocessing.StandardScaler().fit_transform(encoded_output)
    # encoded_output has been scaled to mean=0, std=1


    return (autoencoder.layers[1], encoded_output)


def train_SDAE(hidden_units_list, filename="unlabeled/unlabeled_wifi.h5"):

    X_train = h5py.File(filename, "r")
    for name, i in zip(X_train.keys(), range(len(X_train.keys()))):
        print("shape of unlabeled wifi data:{}".format(np.shape(X_train[name])))
        X = np.array(X_train[name])


    return_list = list()
    input_dim = np.shape(X)[1]
    input_num = np.shape(X)[0]
    factor = 0.1
    # Trains each hidden layer separately (layerwise training of the stacked autoencoder)
    hidden_layer_1, encoded_output = train_layerwise_SDAE(X_train=X,
                                                          input_dim=input_dim,
                                                          hidden_layer_size=hidden_units_list[0],
                                                          noise_factor=factor)
    return_list.append(hidden_layer_1)

    for i, num_hidden_units in enumerate(hidden_units_list[1:]):
        name = 'hidden_layer_{}'.format(i+2)
        # locals() is a built-in function of python, the first returned value has been assigned to a new variables called 'hidden_layer_{i+1}'
        locals()['hidden_layer_{}'.format(i + 2)], encoded_output = train_layerwise_SDAE(X_train=encoded_output,
                                                              input_dim=encoded_output.shape[1],
                                                              hidden_layer_size=num_hidden_units,
                                                              noise_factor=factor)
        return_list.append(eval("hidden_layer_{}".format(i+2)))

    # 根据当前scope存在的变量名,返回相应的hidden_layer,并生成最后的return_list
    # ll = locals().keys()
    # return_name = [v_name for v_name in list(ll) if v_name.startswith("hidden_layer_")]
    # return_list = [globals(name) for name in return_name]

    # for variable in ll:
    #     if variable.startswith("hidden_layer_"):
    #         # variable is a string, the corrresponding varaible should be add to the return_list
    #         # return_list.append(eval(variable))  # version 1
    #         return_list.append(vars()[variable])  # version 2

    return return_list


# trained_SDAE = train_SDAE([200,100,50])

def build_and_finetune_pretrained_classification(pretrained_layer, X_train, Y_train, X_test, Y_test):
    # Creates a deep network consisting of the hidden layer and a classification
    # output layer to perform fine tuning (supervised learning)
    input_layer = Input(shape=(INPUT_DIM,))
    layer1 = pretrained_layer[0](input_layer)
    layer2 = pretrained_layer[1](layer1)
    layer3 = pretrained_layer[2](layer2)
    ouput_layer = Dense(NUM_CLASS, activation="softmax", name="output_layer")(layer3)

    pretrained_network = Model(inputs=input_layer, outputs=ouput_layer)
    pretrained_network.compile(optimizer=sgd(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])

    # Fine tuning
    pretrained_network.fit(X_train, Y_train, epochs=100, batch_size=8, validation_data=[X_test, Y_test])

    return pretrained_network



def build_and_finetune_pretrained_regression(pretrained_layer, X_train, Y_train, X_test, Y_test):
    # Creates a deep network consisting of the hidden layer and a classification
    # output layer to perform fine tuning (supervised learning)
    input_layer = Input(shape=(INPUT_DIM,))
    layer1 = pretrained_layer[0](input_layer)
    layer2 = pretrained_layer[1](layer1)
    layer3 = pretrained_layer[2](layer2)
    ouput_layer = Dense(2, activation="tanh", name="output_layer")(layer3)

    pretrained_network = Model(inputs=input_layer, outputs=ouput_layer)
    pretrained_network.compile(optimizer=sgd(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True), loss='mean_squared_error', metrics=['accuracy'])

    # Fine tuning
    pretrained_network.fit(X_train, Y_train, epochs=200, batch_size=8, validation_data=[X_test, Y_test])

    return pretrained_network