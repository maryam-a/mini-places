import os, datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 25
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.00001
dropout = 0.6 # Dropout, probability to keep units
training_iters = 25000 #370000
step_display = 50 
step_save = 10000
path_save = 'vgg16results_bn'
start_from = ''
CONV_STRIDE = [1,1,1,1]
MAX_POOL_STRIDE = [1,2,2,1]
MAX_POOL_WINDOW = [1,2,2,1]
categories = 100

# File Paths - MODIFY PATHS ACCORDINGLY
## Maryam
data_root = '../data/images/'
data_list_prefix = 'data/'
kerberos = "marchie"

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

def vgg16(x, keep_dropout, train_phase):
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
        'bo': tf.Variable(tf.zeros(categories))
    }
    
    # Convolutional Layer: 3 x 3 window; stride = 1; padding = 1
    # Max Pool: 2 x 2 window; stride = 2
    
    # WL1: Conv (conv3-64) + ReLU
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=CONV_STRIDE, padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    
    # WL2: Conv (conv3-64) + ReLU + Max Pool
    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=CONV_STRIDE, padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    pool1 = tf.nn.max_pool(conv2, ksize=MAX_POOL_WINDOW, strides=MAX_POOL_STRIDE, padding='SAME')

    # WL3: Conv (conv3-128) + ReLU
    conv3 = tf.nn.conv2d(pool1, weights['wc3'], strides=CONV_STRIDE, padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')

    # WL4: Conv (conv3-128) + ReLU + Max Pool
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=CONV_STRIDE, padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    pool2 = tf.nn.max_pool(conv4, ksize=MAX_POOL_WINDOW, strides=MAX_POOL_STRIDE, padding='SAME')

    # WL5: Conv (conv3-256) + ReLU
    conv5 = tf.nn.conv2d(pool2, weights['wc5'], strides=CONV_STRIDE, padding='SAME')
    conv5 = tf.nn.relu(conv5)
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    
    # WL6: Conv (conv3-256) + ReLU
    conv6 = tf.nn.conv2d(conv5, weights['wc6'], strides=CONV_STRIDE, padding='SAME')
    conv6 = tf.nn.relu(conv6)
    conv6 = batch_norm_layer(conv6, train_phase, 'bn6')
    
    # WL7: Conv (conv3-256) + ReLU + Max Pool
    conv7 = tf.nn.conv2d(conv6, weights['wc7'], strides=CONV_STRIDE, padding='SAME')
    conv7 = tf.nn.relu(conv7)
    conv7 = batch_norm_layer(conv7, train_phase, 'bn7')
    pool3 = tf.nn.max_pool(conv7, ksize=MAX_POOL_WINDOW, strides=MAX_POOL_STRIDE, padding='SAME')
    
    # WL8: Conv (conv3-512) + ReLU
    conv8 = tf.nn.conv2d(pool3, weights['wc8'], strides=CONV_STRIDE, padding='SAME')
    conv8 = tf.nn.relu(conv8)
    conv8 = batch_norm_layer(conv8, train_phase, 'bn8')
    
    # WL9: Conv (conv3-512) + ReLU
    conv9 = tf.nn.conv2d(conv8, weights['wc9'], strides=CONV_STRIDE, padding='SAME')
    conv9 = tf.nn.relu(conv9)
    conv9 = batch_norm_layer(conv9, train_phase, 'bn9')
    
    # WL10: Conv (conv3-512) + ReLU + Max Pool
    conv10 = tf.nn.conv2d(conv9, weights['wc10'], strides=CONV_STRIDE, padding='SAME')
    conv10 = tf.nn.relu(conv10)
    conv10 = batch_norm_layer(conv10, train_phase, 'bn10')
    pool4 = tf.nn.max_pool(conv10, ksize=MAX_POOL_WINDOW, strides=MAX_POOL_STRIDE, padding='SAME')
    
    # WL11: Conv (conv3-512) + ReLU
    conv11 = tf.nn.conv2d(pool4, weights['wc11'], strides=CONV_STRIDE, padding='SAME')
    conv11 = tf.nn.relu(conv11)
    conv11 = batch_norm_layer(conv11, train_phase, 'bn11')
    
    # WL12: Conv (conv3-512) + ReLU
    conv12 = tf.nn.conv2d(conv11, weights['wc12'], strides=CONV_STRIDE, padding='SAME')
    conv12 = tf.nn.relu(conv12)
    conv12 = batch_norm_layer(conv12, train_phase, 'bn12')
    
    # WL13: Conv (conv3-512) + ReLU + Max Pool
    conv13 = tf.nn.conv2d(conv12, weights['wc13'], strides=CONV_STRIDE, padding='SAME')
    conv13 = tf.nn.relu(conv13)
    conv13 = batch_norm_layer(conv13, train_phase, 'bn13')
    pool5 = tf.nn.max_pool(conv13, ksize=MAX_POOL_WINDOW, strides=MAX_POOL_STRIDE, padding='SAME')
    
    # WL14: FC + ReLU + Dropout
    fc14 = tf.reshape(pool5, [-1, weights['wc14'].get_shape().as_list()[0]])
    fc14 = tf.matmul(fc14, weights['wc14'])
    fc14 = tf.nn.relu(fc14)
    fc14 = batch_norm_layer(fc14, train_phase, 'bn14')
    fc14 = tf.nn.dropout(fc14, keep_dropout)
    
    # WL15: FC + ReLU + Dropout
    fc15 = tf.matmul(fc14, weights['wc15'])
    fc15 = tf.nn.relu(fc15)
    fc15 = batch_norm_layer(fc15, train_phase, 'bn15')
    fc15 = tf.nn.dropout(fc15, keep_dropout)

    # WL16: Output FC
    fc16 = tf.add(tf.matmul(fc15, weights['wc16']), biases['bo'])
    
    return fc16

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': data_root,   
    'data_list': data_list_prefix + 'train.txt', 
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True,
    'test_data': False
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': data_root,   
    'data_list': data_list_prefix + 'val.txt',   
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'test_data': False
    }
opt_data_test = {
    #'data_h5': 'miniplaces_256_test.h5',
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
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = vgg16(x, keep_dropout, train_phase)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

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

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

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
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.4f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.2f}".format(acc1) + ", Top5 = " + \
                  "{:.2f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.4f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.2f}".format(acc1) + ", Top5 = " + \
                  "{:.2f}".format(acc5))
        
        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
        
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
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.2f}".format(acc1) + ", Top5 = " + \
              "{:.2f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))

    print('Evaluation on the whole test set...')
    output = open(kerberos +'.test.pred.' + datetime.datetime.now().strftime("%Y-%m-%d") + '.txt', 'w')
    test_num_batch = loader_test.size()
    loader_test.reset()

    for j in range(test_num_batch):
        test_images_batch = loader_test.next_batch(1)
        test_image_labels = "test/" + "%08d" % (j+1,) + ".jpg"
        result = sess.run([top5_pred], feed_dict = {x: test_images_batch, keep_dropout: 1., train_phase: False})[0][1][0]
        for l in result:
            test_image_labels = test_image_labels + " " + str(l)
        print (test_image_labels)
        output.write(test_image_labels + "\n")
    output.close()
