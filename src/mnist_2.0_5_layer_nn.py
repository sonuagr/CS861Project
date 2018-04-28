# encoding: UTF-8
# Copyright Krzysztof Sopyła (krzysztofsopyla@gmail.com)
#
#
# Licensed under the MIT

# Network architecture:
# Five layer neural network, input layer 28*28= 784, output 10 (10 digits)
# Output labels uses one-hot encoding

# input layer             - X[batch, 784]
# 1 layer                 - W1[784, 200] + b1[200]
#                           Y1[batch, 200] 
# 2 layer                 - W2[200, 100] + b2[100]
#                           Y2[batch, 200] 
# 3 layer                 - W3[100, 60]  + b3[60]
#                           Y3[batch, 200] 
# 4 layer                 - W4[60, 30]   + b4[30]
#                           Y4[batch, 30] 
# 5 layer                 - W5[30, 10]   + b5[10]
# One-hot encoded labels    Y5[batch, 10]

# model
# Y = softmax(X*W+b)
# Matrix mul: X*W - [batch,784]x[784,10] -> [batch,10]

# Training consists of finding good W elements. This will be handled automaticaly by 
# Tensorflow optimizer


# import visualizations as vis
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from numpy import genfromtxt


NUM_ITERS=5000
DISPLAY_STEP=100
BATCH=100

tf.set_random_seed(0)

# Download images and labels 
mnist = read_data_sets("MNISTdata", one_hot=True, reshape=False, validation_size=0)

# mnist.test (10K images+labels) -> mnist.test.images, mnist.test.labels
# mnist.train (60K images+labels) -> mnist.train.images, mnist.test.labels

# Placeholder for input images, each data sample is 28x28 grayscale images
# All the data will be stored in X - tensor, 4 dimensional matrix
# The first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])


# layers sizes
# L1 = 200
# L2 = 100
# L3 = 60
# L4 = 30
# L5 = 10

# layers sizes
L1 = 200
L2 = 1000
L3 = 600
L4 = 30
L5 = 10

# weights - initialized with random values from normal distribution mean=0, stddev=0.1
# output of one layer is input for the next

# W1_data = genfromtxt('W1_val.csv', delimiter=',').astype(np.float32)
# W2_data = genfromtxt('W2_val.csv', delimiter=',').astype(np.float32)
# W3_data = genfromtxt('W3_val.csv', delimiter=',').astype(np.float32)
# W4_data = genfromtxt('W4_val.csv', delimiter=',').astype(np.float32)
# W5_data = genfromtxt('W5_val.csv', delimiter=',').astype(np.float32)

# b1_data = genfromtxt('b1_val.csv', delimiter=',').astype(np.float32)
# b2_data = genfromtxt('b2_val.csv', delimiter=',').astype(np.float32)
# b3_data = genfromtxt('b3_val.csv', delimiter=',').astype(np.float32)
# b4_data = genfromtxt('b4_val.csv', delimiter=',').astype(np.float32)
# b5_data = genfromtxt('b5_val.csv', delimiter=',').astype(np.float32)


# W1 = tf.Variable(W1_data)
# W2 = tf.Variable(W2_data)
# W3 = tf.Variable(W3_data)
# W4 = tf.Variable(W4_data)
# W5 = tf.Variable(W5_data)

# b1 = tf.Variable(b1_data)
# b2 = tf.Variable(b2_data)
# b3 = tf.Variable(b3_data)
# b4 = tf.Variable(b4_data)
# b5 = tf.Variable(b5_data)


W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
b1 = tf.Variable(tf.zeros([L1]))

W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
b2 = tf.Variable(tf.zeros([L2]))

W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
b3 = tf.Variable(tf.zeros([L3]))

W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
b4 = tf.Variable(tf.zeros([L4]))

W5 = tf.Variable(tf.truncated_normal([L4, L5], stddev=0.1))
b5 = tf.Variable(tf.zeros([L5]))


# flatten the images, unrole eacha image row by row, create vector[784]
# -1 in the shape definition means compute automatically the size of this dimension
XX = tf.reshape(X, [-1, 784])

# Define model
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + b1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + b2)
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + b3)
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + b4)
Ylogits = tf.matmul(Y4, W5) + b5
Y = tf.nn.softmax(Ylogits)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
#cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 100.0  # normalized for batches of 100 images,


# we can also use tensorflow function for softmax
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

                                                          
# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(b1, [-1]), tf.reshape(b2, [-1]), tf.reshape(b3, [-1]), tf.reshape(b4, [-1]), tf.reshape(b5, [-1])], 0)




# Initializing the variables
init = tf.global_variables_initializer()

train_losses = list()
train_acc = list()
test_losses = list()
test_acc = list()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for i in range(NUM_ITERS+1):
        # training on batches of 100 images with 100 labels
        batch_X, batch_Y = mnist.train.next_batch(BATCH)
        
        if i%DISPLAY_STEP ==0:
            # compute training values for visualisation
            acc_trn, loss_trn, w, b = sess.run([accuracy, cross_entropy, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
            
            
            acc_tst, loss_tst = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
            
            print("#{} Trn acc={} , Trn loss={} Tst acc={} , Tst loss={}".format(i,acc_trn,loss_trn,acc_tst,loss_tst))

            train_losses.append(loss_trn)
            train_acc.append(acc_trn)
            test_losses.append(loss_tst)
            test_acc.append(acc_tst)

        # the backpropagationn training step
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})
        # saver.save(sess, 'my_model')

    # W1_val, b1_val, W2_val, b2_val, W3_val, b3_val, W4_val, b4_val, W5_val, b5_val = sess.run([W1, b1, W2, b2, W3, b3, W4, b4, W5, b5])

    # np.savetxt("W1_val.csv", W1_val, delimiter=",")
    # np.savetxt("b1_val.csv", b1_val, delimiter=",")
    # np.savetxt("W2_val.csv", W2_val, delimiter=",")
    # np.savetxt("b2_val.csv", b2_val, delimiter=",")
    # np.savetxt("W3_val.csv", W3_val, delimiter=",")
    # np.savetxt("b3_val.csv", b3_val, delimiter=",")
    # np.savetxt("W4_val.csv", W4_val, delimiter=",")
    # np.savetxt("b4_val.csv", b4_val, delimiter=",")
    # np.savetxt("W5_val.csv", W5_val, delimiter=",")
    # np.savetxt("b5_val.csv", b5_val, delimiter=",")

# title = "MNIST 2.0 5 layers sigmoid"
# vis.losses_accuracies_plots(train_losses,train_acc,test_losses, test_acc,title,DISPLAY_STEP)






# Restults
# mnist_single_layer_nn.py acc= 0.9237 
# mnist__layer_nn.py TST acc = 0.9534


# sample output for 5k iterations 
#0 Trn acc=0.10999999940395355 , Trn loss=230.5011444091797 Tst acc=0.0957999974489212 , Tst loss=232.8909912109375
#100 Trn acc=0.10000000149011612 , Trn loss=229.38812255859375 Tst acc=0.09799999743700027 , Tst loss=230.8378448486328
#200 Trn acc=0.07000000029802322 , Trn loss=231.29209899902344 Tst acc=0.09799999743700027 , Tst loss=230.82485961914062
#300 Trn acc=0.09000000357627869 , Trn loss=232.11734008789062 Tst acc=0.10090000182390213 , Tst loss=230.51341247558594
# ...
#4800 Trn acc=0.949999988079071 , Trn loss=11.355264663696289 Tst acc=0.948199987411499 , Tst loss=17.340219497680664
#4900 Trn acc=0.9399999976158142 , Trn loss=22.300941467285156 Tst acc=0.9466999769210815 , Tst loss=17.51348876953125
#5000 Trn acc=0.9200000166893005 , Trn loss=20.947153091430664 Tst acc=0.953499972820282 , Tst loss=15.77566909790039
