import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data

database = input_data.read_data_sets('.\content\data', one_hot=True)
tf.reset_default_graph()
# Parameters for neural network
LEARNING_PARAM = 0.001
NUMBER_OF_ITERATIONS = 3000
BATCH_SIZE = 32
image_dimension = 784  # image size is 28 x 28
NUMBER_OF_NEURONS = 256
LATENT_VARIABLE_DIMENSION = 10 #5
# with attention or without
WITH_ATTENTION = True

T_RANGE = 10 #3
SHARE_PARAMETERS = False

IMAGE_SIZE = 28
IMG_TYPE_TRAINING = 'TRAIN'
IMG_TYPE_TESTING = 'TEST'
ATTENTION_GRID_SIZE_N = 5

#initialize tf variables
images = tf.placeholder(tf.float32, shape=[None, image_dimension])

random_noise = tf.random_normal((BATCH_SIZE, LATENT_VARIABLE_DIMENSION), mean=0, stddev=1)  # Qsampler noise

lstm_enc = tf.nn.rnn_cell.LSTMCell(NUMBER_OF_NEURONS, state_is_tuple=True)
lstm_dec = tf.nn.rnn_cell.LSTMCell(NUMBER_OF_NEURONS, state_is_tuple=True)

h_t_dec_prev = tf.zeros((BATCH_SIZE, NUMBER_OF_NEURONS))

c = [0] * T_RANGE
mu = [0] * T_RANGE
logsigma = [0] * T_RANGE
sigma = [0] * T_RANGE

h_t_enc = lstm_enc.zero_state(BATCH_SIZE, tf.float32)
dec_state = lstm_dec.zero_state(BATCH_SIZE, tf.float32)

def encode_RNN(prev_state, image):
    with tf.variable_scope("encoder", reuse=SHARE_PARAMETERS):
        hidden_layer, next_state = lstm_enc(image, prev_state)
    with tf.variable_scope("mu", reuse=SHARE_PARAMETERS):
        mu = apply_weights(hidden_layer, NUMBER_OF_NEURONS, LATENT_VARIABLE_DIMENSION)
    with tf.variable_scope("sigma", reuse=SHARE_PARAMETERS):
        logsigma = apply_weights(hidden_layer, NUMBER_OF_NEURONS, LATENT_VARIABLE_DIMENSION)
        sigma = tf.exp(logsigma)
    return mu, logsigma, sigma, next_state

def apply_weights(hidden_layer, inputFeatures, outputFeatures):
    weights = tf.get_variable("weights", [inputFeatures, outputFeatures], tf.float32,  tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
    return tf.matmul(hidden_layer, weights) + bias

def decode_RNN(prev_state, latent):
    with tf.variable_scope("decoder", reuse=SHARE_PARAMETERS):
        hidden_layer, next_state = lstm_dec(latent, prev_state)

    return hidden_layer, next_state

def write(hidden_layer):
    if WITH_ATTENTION:
        return write_with_attention(hidden_layer)
    else:
        return write_without_attention(hidden_layer)

def write_without_attention(hidden_layer):
    with tf.variable_scope("write", reuse=SHARE_PARAMETERS):
        decoded_image_portion = apply_weights(hidden_layer, NUMBER_OF_NEURONS, image_dimension)
    return decoded_image_portion


def read(x, error_image_t, h_t_dec_prev):
    if WITH_ATTENTION:
        return read_with_attention(x, error_image_t, h_t_dec_prev)
    else:
        return read_without_attention(x, error_image_t)

def read_without_attention(x, error_image_t):
    return tf.concat([x, error_image_t], 1)  # read without attention = concat

def filter_img(img, Fx, Fy, gamma):
    Fx_t = tf.transpose(Fx, perm=[0, 2, 1])
    img = tf.reshape(img, [-1, IMAGE_SIZE, IMAGE_SIZE])
    glimpse = tf.matmul(Fy, tf.matmul(img, Fx_t))
    glimpse = tf.reshape(glimpse, [-1, ATTENTION_GRID_SIZE_N ** 2])
    return glimpse * tf.reshape(gamma, [-1, 1])

def read_with_attention(x, error_image_t, h_t_dec_prev):
    Fx, Fy, gamma = linear_tranformation_of_decoder_output("read", h_t_dec_prev)
    x = filter_img(x, Fx, Fy, gamma)
    x_hat = filter_img(error_image_t, Fx, Fy, gamma)
    return tf.concat([x, x_hat], 1)

def write_with_attention(hidden_layer):
    with tf.variable_scope("writeW", reuse=SHARE_PARAMETERS):
        # [Equation 28]
        w = apply_weights(hidden_layer, NUMBER_OF_NEURONS, ATTENTION_GRID_SIZE_N ** 2)
    w = tf.reshape(w, [BATCH_SIZE, ATTENTION_GRID_SIZE_N, ATTENTION_GRID_SIZE_N])
    Fx, Fy, gamma = linear_tranformation_of_decoder_output("write", hidden_layer)

    # [Equation 29 - part]
    wr_tmp = tf.matmul(tf.transpose(Fy, perm=[0, 2, 1]), tf.matmul(w, Fx))
    wr = tf.reshape(wr_tmp, [BATCH_SIZE, IMAGE_SIZE ** 2])
    # [Equation 29]
    return wr * tf.reshape(1.0 / gamma, [-1, 1])


def linear_tranformation_of_decoder_output(scope, h_dec):
    with tf.variable_scope(scope, reuse=SHARE_PARAMETERS):
        parameters = apply_weights(h_dec, NUMBER_OF_NEURONS, 5)
    # [Equation 21]
    gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(parameters, 5, 1)

    # [Equation 22]
    gx = ((IMAGE_SIZE + 1) / 2) * (gx_ + 1)
    # [Equation 23]
    gy = ((IMAGE_SIZE + 1) / 2) * (gy_ + 1)

    sigma2 = tf.exp(log_sigma2)
    gamma = tf.exp(log_gamma)

    # [Equation 24]
    delta = (IMAGE_SIZE - 1) / ((ATTENTION_GRID_SIZE_N - 1) * tf.exp(log_delta))

    Fx, Fy= calculate_filterbank_matrices(gx, gy, sigma2, delta)
    return [Fx, Fy, gamma]

def calculate_filterbank_matrices(gx, gy, sigma2, delta):
    grid_i = tf.reshape(tf.cast(tf.range(ATTENTION_GRID_SIZE_N), tf.float32), [1, -1])

    # [Equation 19, 20]
    mu_x = tf.reshape(gx + (grid_i - ATTENTION_GRID_SIZE_N / 2 - 0.5) * delta, [-1, ATTENTION_GRID_SIZE_N, 1])
    mu_y = tf.reshape(gy + (grid_i - ATTENTION_GRID_SIZE_N / 2 - 0.5) * delta, [-1, ATTENTION_GRID_SIZE_N, 1])

    a_b = tf.reshape(tf.cast(tf.range(IMAGE_SIZE), tf.float32), [1, 1, -1])

    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    # [Equation 25, 26]
    fx_tmp = tf.exp(-tf.square(a_b - mu_x) / (2 * sigma2))
    fy_tmp = tf.exp(-tf.square(a_b - mu_y) / (2 * sigma2))

    # [Equation 25, 26] - normalize
    Fx = fx_tmp / tf.maximum(tf.reduce_sum(fx_tmp, 2, keep_dims=True), 1e-8)
    Fy = fy_tmp / tf.maximum(tf.reduce_sum(fy_tmp, 2, keep_dims=True), 1e-8)
    return Fx, Fy


def loss_function(generated_images):
    l_x = tf.reduce_mean(-tf.reduce_sum(images * tf.log(1e-10 + generated_images) + (1 - images) * tf.log(1e-10 + 1 - generated_images), 1))
    kl = [0] * T_RANGE
    for iter in range(T_RANGE):
        mu2 = tf.square(mu[iter])
        sigma2 = tf.square(sigma[iter])
        logsigma_X = logsigma[iter]
        # log (sigma_X **2) = 2 * log(sigma_X)
        kl[iter] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma_X, 1) - T_RANGE * 0.5
    l_z = tf.reduce_mean(tf.add_n(kl))
    return l_x + l_z

def show_and_save_results(image, type):
    # show results
    plt.figure(figsize=(8, 10))
    plt.imshow(empty_image, origin="upper", cmap="gray")
    plt.grid(False)
    plt.show()

    # save results
    img_type = "TRAINING_" if type == 'TRAIN' else "TESTING_"
    prefix = "with_attention" if WITH_ATTENTION else "without_attention"
    os.makedirs("results", exist_ok=True)
    filename = ".\\results\\" + img_type + prefix + "_" + str(NUMBER_OF_ITERATIONS) + "_epochs_" + str(datetime.now().strftime("%d-%m-%Y___%H-%M-%S")) + ".png"
    plt.imsave(fname=filename, arr=image, origin="upper", cmap="gray")

# --- TRAIN -----
x = images
for t in range(T_RANGE):
    if t == 0:
        c_prev = tf.zeros((BATCH_SIZE, image_dimension))
    else:
        c_prev = c[t - 1]

    #c_prev = tf.layers.batch_normalization(c_prev)
    error_image_t = x - tf.sigmoid(c_prev)
    r_t = read(x, error_image_t, h_t_dec_prev)
    mu[t], logsigma[t], sigma[t], h_t_enc = encode_RNN(h_t_enc, tf.concat([r_t, h_t_dec_prev], 1))
    z_t = mu[t] + sigma[t] * random_noise
    h_t_dec, dec_state = decode_RNN(dec_state, z_t)
    c[t] = c_prev + write(h_t_dec)
    h_t_dec_prev = h_t_dec
    SHARE_PARAMETERS = True

# ---- CALCULATE LOSS FUNCTION AND OPTIMIZE ---
#c[-1] = tf.layers.batch_normalization(c[-1])
generated_image = tf.nn.sigmoid(c[-1])
loss_value = loss_function(generated_image)
optimizer = tf.train.AdamOptimizer(LEARNING_PARAM, beta1=0.5).minimize(loss_value)

# Initialize all variables and executing the computational graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


n = 10
stride_x = 0
stride_y = 0
empty_image = np.empty((IMAGE_SIZE * n, IMAGE_SIZE * n))
step = int(NUMBER_OF_ITERATIONS / (n * n))

# Train neural network
for i in range(NUMBER_OF_ITERATIONS):
    x_batch, _ = database.train.next_batch(BATCH_SIZE)
    loss, _, cs = sess.run([loss_value, optimizer, c], feed_dict={images: x_batch})
    if i % step == 0:

        result = 1.000 / (1.000 + np.exp(-np.array(cs)))
        empty_image[(n - stride_x - 1) * IMAGE_SIZE: (n - stride_x) * IMAGE_SIZE, stride_y * IMAGE_SIZE: (stride_y + 1) * IMAGE_SIZE] = result[-1][0].reshape(IMAGE_SIZE, IMAGE_SIZE)
        stride_y += 1
        if stride_y == 10:
            stride_y = 0
            stride_x += 1
        print("Loss is {0} at iteration {1}".format(loss, i))

show_and_save_results(empty_image, IMG_TYPE_TRAINING)


#  --- TESTING ----
##noise_X = tf.placeholder(tf.float32, shape=[None, LATENT_VARIABLE_DIMENSION])
dec_state = lstm_dec.zero_state(BATCH_SIZE, tf.float32)
c_test = [0] * T_RANGE
noise_X = tf.random_normal((BATCH_SIZE, LATENT_VARIABLE_DIMENSION), mean=0, stddev=1)
for t in range(T_RANGE):
    if t == 0:
        c_prev = tf.zeros((BATCH_SIZE, image_dimension))
    else:
        c_prev = c_test[t - 1]
    h_t_dec, dec_state = decode_RNN(dec_state, noise_X)
    c_test[t] = c_prev + write(h_t_dec)
    h_t_dec_prev = h_t_dec
    SHARE_PARAMETERS = True
    noise_X = tf.random_normal((BATCH_SIZE, LATENT_VARIABLE_DIMENSION), mean=0, stddev=1)


# Showing output
n = 10
empty_image = np.empty((IMAGE_SIZE * n, IMAGE_SIZE * n))

for i in range(n):
    for j in range(n):
        noise = np.random.normal(0, 1, size=[BATCH_SIZE, LATENT_VARIABLE_DIMENSION])
        #noise = tf.random_normal((batch_size, latent_variable_dimension), mean=0, stddev=1)
        x_batch = np.random.normal(0, 1, size=[BATCH_SIZE, image_dimension])

        cs = sess.run(c_test, feed_dict={images: x_batch})
        result = 1.000 / (1.000 + np.exp(-np.array(cs)))
        empty_image[(n - i - 1) * IMAGE_SIZE: (n - i) * IMAGE_SIZE, j * IMAGE_SIZE: (j + 1) * IMAGE_SIZE] = result[-1][0].reshape(IMAGE_SIZE, IMAGE_SIZE)


show_and_save_results(empty_image, IMG_TYPE_TESTING)

# TENSORBOARD SECTION
#writer = tf.summary.FileWriter('./graph_example', sess.graph)
#tensorboard --logdir="graph_example"
#writer.close()

sess.close()
