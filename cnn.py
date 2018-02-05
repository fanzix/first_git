import tensorflow as tf
import numpy as np
import random

images=np.load("dataset/44*44_dataset/training_images.npy")
labels=np.load("dataset/44*44_dataset/training_labels.npy")

validation_images=np.load("dataset/44*44_dataset/validation_images.npy")
validation_labels=np.load("dataset/44*44_dataset/validation_labels.npy")

test_images=np.load("dataset/44*44_dataset/test_images.npy")
test_labels=np.load("dataset/44*44_dataset/test_labels.npy")

# images=np.load("dataset/data_augmentation/aug_training_images.npy")
# labels=np.load("dataset/data_augmentation/aug_training_labels.npy")

# validation_images=np.load("dataset/data_augmentation/aug_validation_images.npy")
# validation_labels=np.load("dataset/data_augmentation/aug_validation_labels.npy")

# test_images=np.load("dataset/data_augmentation/aug_training_images.npy")
# test_labels=np.load("dataset/data_augmentation/aug_training_labels.npy")

x = tf.placeholder(tf.float32, [None, 44, 44, 1])
y = tf.placeholder(tf.float32, [None, 2])
prob = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
conv1 = tf.nn.conv2d(input = x, filter=w1, padding="SAME", strides=[1, 1, 1, 1])
conv1 = tf.nn.relu(conv1)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
conv2 = tf.nn.conv2d(pool1, filter=w2, strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.relu(conv2)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

pool3_flat = tf.reshape(pool2, [-1, 11 * 11 * 64])
dense = tf.layers.dense(pool3_flat, units = 128, activation=tf.nn.relu)
dense = tf.nn.dropout(dense, prob)
model = tf.layers.dense(dense,units = 2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(len(images) / batch_size)

data = list(zip(images,labels))

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

for epoch in range(20):
    image = []
    label = []
    random.shuffle(data)
    image[:],label[:] = zip(*data)
    total_cost = 0
    image = np.array(image)
    label = np.array(label)

    for i in range(total_batch):

        a = i * batch_size
        b = (i + 1) * batch_size

        batch_xs, batch_ys = image[a:b], label[a:b]
        batch_xs = batch_xs.reshape(-1,44, 44, 1)
       
        _, cost_val = sess.run([optimizer, cost],feed_dict={x: batch_xs, y: batch_ys, prob:0.5})
        total_cost += cost_val

    print("Epoch {0}: cost = {1}".format(epoch+1, round(total_cost / total_batch, 5)))
    print('accuracy of validation dataset = {0}'.format(sess.run(accuracy, feed_dict={x: validation_images.reshape(-1, 44, 44, 1), y: validation_labels, prob:1})))

print('accuracy of test dataset = {0}'.format(sess.run(accuracy, feed_dict={x: test_images.reshape(-1, 44, 44, 1), y: test_labels, prob:1})))




