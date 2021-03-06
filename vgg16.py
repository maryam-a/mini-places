#Authors - Sandeep Silwal and Maryam Archie
import os, datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from DataLoader import *

# Dataset Parameters
batch_size = 25
load_size = 256
fine_size = 224
c = 3
# recomputed for new dataset
data_mean = np.asarray([0.44947848234408205, 0.43278461113723926, 0.40700250452738196])

# Training Parameters
learning_rate = 0.0001
dropout = 0.6
training_iters = 100000
step_display = 50 
step_save = 10000
path_save = 'vgg16results'
start_from = ''
CONV_STRIDE = [1,1,1,1]
MAX_POOL_STRIDE = [1,2,2,1]
MAX_POOL_WINDOW = [1,2,2,1]
categories = 100

# File Paths - MODIFY PATHS ACCORDINGLY
data_root = '../data/images/'
data_list_prefix = 'data/'
training_file = 'train.txt' # 100,000 -> old_train.txt; 400,000 -> train.txt

team_name = 'visionaries'

def vgg16(x, keep_dropout):
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=np.sqrt(2./(3*3*3)))),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2./(3*3*64)))),
                                          
        'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=np.sqrt(2./(3*3*64)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2./(3*3*128)))),
                                          
        'wc5': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=np.sqrt(2./(3*3*128)))),
        'wc6': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
        'wc7': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
                                         
        'wc8': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=np.sqrt(2./(3*3*256)))),  
        'wc9': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))), 
        'wc10': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))), 
    
        'wc11': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))), 
        'wc12': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))), 
        'wc13': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))), 
   
        'wc14': tf.Variable(tf.random_normal([7*7*512, 4096], stddev=np.sqrt(2./(7*7*512)))),
        'wc15': tf.Variable(tf.random_normal([1*1*4096, 4096], stddev=np.sqrt(2./(1*1*4096)))),
        'wc16': tf.Variable(tf.random_normal([1*1*4096, categories], stddev=np.sqrt(2./(1*1*4096))))
    }

    biases = {
        'bc1': tf.Variable(tf.zeros(64)),
        'bc2': tf.Variable(tf.zeros(64)),
        'bc3': tf.Variable(tf.zeros(128)),
        'bc4': tf.Variable(tf.zeros(128)),
        'bc5': tf.Variable(tf.zeros(256)),
        'bc6': tf.Variable(tf.zeros(256)),
        'bc7': tf.Variable(tf.zeros(256)),
        'bc8': tf.Variable(tf.zeros(512)),
        'bc9': tf.Variable(tf.zeros(512)),
        'bc10': tf.Variable(tf.zeros(512)),
        'bc11': tf.Variable(tf.zeros(512)),
        'bc12': tf.Variable(tf.zeros(512)),
        'bc13': tf.Variable(tf.zeros(512)),
        'bc14': tf.Variable(tf.zeros(4096)),
        'bc15': tf.Variable(tf.zeros(4096)),
        'bc16': tf.Variable(tf.zeros(categories))
    }

    # Convolutional Layer: 3 x 3 window; stride = 1; padding = 1
    # Max Pool: 2 x 2 window; stride = 2

    # WL1: Conv (conv3-64) + ReLU
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=CONV_STRIDE, padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))

    # WL2: Conv (conv3-64) + ReLU + Max Pool
    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=CONV_STRIDE, padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
    pool1 = tf.nn.max_pool(conv2, ksize=MAX_POOL_WINDOW, strides=MAX_POOL_STRIDE, padding='SAME')

    # WL3: Conv (conv3-128) + ReLU
    conv3 = tf.nn.conv2d(pool1, weights['wc3'], strides=CONV_STRIDE, padding='SAME')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']))

    # WL4: Conv (conv3-128) + ReLU + Max Pool
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=CONV_STRIDE, padding='SAME')
    conv4 = tf.nn.relu(tf.nn.bias_add(conv4, biases['bc4']))
    pool2 = tf.nn.max_pool(conv4, ksize=MAX_POOL_WINDOW, strides=MAX_POOL_STRIDE, padding='SAME')

    # WL5: Conv (conv3-256) + ReLU
    conv5 = tf.nn.conv2d(pool2, weights['wc5'], strides=CONV_STRIDE, padding='SAME')
    conv5 = tf.nn.relu(tf.nn.bias_add(conv5, biases['bc5']))

    # WL6: Conv (conv3-256) + ReLU
    conv6 = tf.nn.conv2d(conv5, weights['wc6'], strides=CONV_STRIDE, padding='SAME')
    conv6 = tf.nn.relu(tf.nn.bias_add(conv6, biases['bc6']))

    # WL7: Conv (conv3-256) + ReLU + Max Pool
    conv7 = tf.nn.conv2d(conv6, weights['wc7'], strides=CONV_STRIDE, padding='SAME')
    conv7 = tf.nn.relu(tf.nn.bias_add(conv7, biases['bc7']))
    pool3 = tf.nn.max_pool(conv7, ksize=MAX_POOL_WINDOW, strides=MAX_POOL_STRIDE, padding='SAME')

    # WL8: Conv (conv3-512) + ReLU
    conv8 = tf.nn.conv2d(pool3, weights['wc8'], strides=CONV_STRIDE, padding='SAME')
    conv8 = tf.nn.relu(tf.nn.bias_add(conv8, biases['bc8']))

    # WL9: Conv (conv3-512) + ReLU
    conv9 = tf.nn.conv2d(conv8, weights['wc9'], strides=CONV_STRIDE, padding='SAME')
    conv9 = tf.nn.relu(tf.nn.bias_add(conv9, biases['bc9']))

    # WL10: Conv (conv3-512) + ReLU + Max Pool
    conv10 = tf.nn.conv2d(conv9, weights['wc10'], strides=CONV_STRIDE, padding='SAME')
    conv10 = tf.nn.relu(tf.nn.bias_add(conv10, biases['bc10']))
    pool4 = tf.nn.max_pool(conv10, ksize=MAX_POOL_WINDOW, strides=MAX_POOL_STRIDE, padding='SAME')

    # WL11: Conv (conv3-512) + ReLU
    conv11 = tf.nn.conv2d(pool4, weights['wc11'], strides=CONV_STRIDE, padding='SAME')
    conv11 = tf.nn.relu(tf.nn.bias_add(conv11, biases['bc11']))

    # WL12: Conv (conv3-512) + ReLU
    conv12 = tf.nn.conv2d(conv11, weights['wc12'], strides=CONV_STRIDE, padding='SAME')
    conv12 = tf.nn.relu(tf.nn.bias_add(conv12, biases['bc12']))

    # WL13: Conv (conv3-512) + ReLU + Max Pool
    conv13 = tf.nn.conv2d(conv12, weights['wc13'], strides=CONV_STRIDE, padding='SAME')
    conv13 = tf.nn.relu(tf.nn.bias_add(conv13, biases['bc13']))
    pool5 = tf.nn.max_pool(conv13, ksize=MAX_POOL_WINDOW, strides=MAX_POOL_STRIDE, padding='SAME')

    # WL14: FC + ReLU + Dropout
    fc14 = tf.reshape(pool5, [-1, weights['wc14'].get_shape().as_list()[0]])
    fc14 = tf.add(tf.matmul(fc14, weights['wc14']), biases['bc14'])
    fc14 = tf.nn.relu(fc14)
    fc14 = tf.nn.dropout(fc14, keep_dropout)

    # WL15: FC + ReLU + Dropout
    fc15 = tf.add(tf.matmul(fc14, weights['wc15']), biases['bc15'])
    fc15 = tf.nn.relu(fc15)
    fc15 = tf.nn.dropout(fc15, keep_dropout)

    # WL16: Output FC
    fc16 = tf.add(tf.matmul(fc15, weights['wc16']), biases['bc16'])

    return fc16

# Construct dataloader
opt_data_train = {
    'data_root': data_root,   
    'data_list': data_list_prefix + training_file, 
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True,
    'test_data': False
    }

opt_data_val = {
    'data_root': data_root,   
    'data_list': data_list_prefix + 'val.txt',   
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'test_data': False
    }

opt_data_test = {
    'data_root': data_root,   
    'data_list': data_list_prefix + 'test.txt',   
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'test_data': True
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
loader_test = DataLoaderDisk(**opt_data_test)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)

# Construct model
logits = vgg16(x, keep_dropout)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
train_optimizer2 = tf.train.AdamOptimizer(learning_rate=(learning_rate/10.)).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))
top5_pred = tf.nn.top_k(logits, 5)

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# ensure save location is present
if not Path(path_save).is_dir():
    Path(path_save).mkdir()

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)

    step = 0

    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)

        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.}) 
            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.4f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.2f}".format(acc1) + ", Top5 = " + \
                  "{:.2f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1.}) 
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.4f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.2f}".format(acc1) + ", Top5 = " + \
                  "{:.2f}".format(acc5))

        # Run optimization op (backprop)
        if step < 50000:
            sess.run(train_optimizer1, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout})
        else:            
            sess.run(train_optimizer2, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout})

        step += 1

        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))

    print("Optimization Finished!")

    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)    
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.2f}".format(acc1) + ", Top5 = " + \
              "{:.2f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))

    # Evaluate on the whole test set
    print('Evaluation on the whole test set...')
    output = open(team_name +'.test.pred.' + datetime.datetime.now().strftime("%Y-%m-%d") + '.txt', 'w')
    test_num_batch = loader_test.size()
    loader_test.reset()

    for j in range(test_num_batch):
        test_images_batch = loader_test.next_batch(1)
        test_image_labels = "test/" + "%08d" % (j+1,) + ".jpg"
        result = sess.run([top5_pred], feed_dict = {x: test_images_batch, keep_dropout: 1.})[0][1][0]
        for l in result:
            test_image_labels = test_image_labels + " " + str(l)
        print (test_image_labels)
        output.write(test_image_labels + "\n")
    output.close()
