from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
import Plotting


'''
# ---------------------------- 1. construct a simple autoencoder ----------------------------------
encoding_dim = 32

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)  # 给该层加入l1正则化，进行权重的的稀疏约束(使大部分权重为0？)
encoded = Dense(64, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(encoded)  # 给该层加入l1正则化，进行权重的的稀疏约束(使大部分权重为0？)
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(encoded)  # 给该层加入l1正则化，进行权重的的稀疏约束(使大部分权重为0？)
# 加入regularizer使其更不容易过拟合，但是训练时间会延长
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# construct the 'auto_encoder'
autoencoder = Model(input_img, decoded)

# construct the 'encoder'
encoder = Model(input_img, encoded)

# construct the 'decoder'
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))    # 注意这里的第二个参数（应该是不能用Decoded替代）

# 模型构造总结，输入用Input, 层用Dense, Dense的参数第一个为该层输出的维度，第二个为激活函数
# Model函数的两个参数，第一个为输入层Input，第二个为整个网络结构，即到最后输出的样子Dense、

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

# training the autoencoder
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=0)


#-------------------------- 2. visualization ---------------------------

# using the real test data feed to the separated models 'encoder' and 'decoder' to predict,
# and get the 'encoded_imgs' and 'decoded_imgs'
Plotting.plot_train_val(autoencoder.history.history, "Mnist")

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))   # autoencoder的输入/encoder的输入
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))    # autoencoder的输出/decoder的输出
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
'''




# ------------------------ ** Sequential data examples ** -------------------------

# from keras.layers import Input, LSTM, RepeatVector
# from keras.models import Model
#
# inputs = Input(shape=(timesteps, input_dim))
# encoded = LSTM(latent_dim)(inputs)
#
# decoded = RepeatVector(timesteps)(encoded)    # 此处重复的次数应该与输出序列的时间步数一致，在此例中直接写timesteps就是说输出的时间步和输入的时间步都是timesteps
# decoded = LSTM(input_dim, return_sequences=True)(decoded)
#
# sequence_autoencoder = Model(inputs, decoded)
# encoder = Model(inputs, encoded)

# ------------------------ ** Variational autoencoder examples ** -------------------------

# x = Input(batch_shape=(batch_size, original_dim))
# h = Dense(intermediate_dim, activation='relu')(x)
# z_mean = Dense(latent_dim)(h)
# z_log_sigma = Dense(latent_dim)(h)
#
# def sampling(args):
#     z_mean, z_log_sigma = args
#     epsilon = K.random_normal(shape=(batch_size, latent_dim),
#                               mean=0., std=epsilon_std)
#     return z_mean + K.exp(z_log_sigma) * epsilon
#
# # note that "output_shape" isn't necessary with the TensorFlow backend
# # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
# z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
#
#
# decoder_h = Dense(intermediate_dim, activation='relu')
# decoder_mean = Dense(original_dim, activation='sigmoid')
# h_decoded = decoder_h(z)
# x_decoded_mean = decoder_mean(h_decoded)
#
#
# # 1.end-to-end autoencoder
# vae = Model(x, x_decoded_mean)
#
# # 2.encoder, from inputs to latent space
# encoder = Model(x, z_mean)
#
# # 3.generator, from latent space to reconstructed inputs
# decoder_input = Input(shape=(latent_dim,))
# _h_decoded = decoder_h(decoder_input)
# _x_decoded_mean = decoder_mean(_h_decoded)
# generator = Model(decoder_input, _x_decoded_mean)
#
#
# def vae_loss(x, x_decoded_mean):
#     xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
#     kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
#     return xent_loss + kl_loss
#
# vae.compile(optimizer='rmsprop', loss=vae_loss)
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#
# vae.fit(x_train, x_train,
#         shuffle=True,
#         epochs=epochs,
#         batch_size=batch_size,
#         validation_data=(x_test, x_test))



# --------------------------- test --------------------------------

# n=4
# grid_x = np.linspace(-15, 15, n)
# grid_y = np.linspace(-15, 15, n)
#
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         print("i:{},\tyi:{},\tj:{},\txi:{},\t".format(i, yi, j, xi))







