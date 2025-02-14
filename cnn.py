import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

def load_numpy(var_name):
	return np.load("pretrained_weights/"+var_name+".npy")

def save_numpy(tfarr, var_name):
	np.save("pretrained_weights/"+var_name+".npy", tfarr)

def rev_one_hot(arr):
	for e in range(len(arr)):
		if arr[e] == 1:
			return e

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W, strides=[1, 1, 1, 1]):
	return tf.nn.conv2d(x, W, strides, padding='SAME')

def max_pool_2x2(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
	return tf.nn.max_pool(x, ksize, strides, padding='SAME')

def create_model((train_data_x, train_data_y), (test_data_x, test_data_y), num_conv_layers=2, num_fc_layers=2):
	sess = tf.Session()
	x_size = train_data_x[0].size #reduce(lambda x, y: x*y, trainDataX[0].size)
	y_size = len(train_data_y[0])
	x = tf.placeholder(tf.float32, shape=[None, x_size])
	y_ = tf.placeholder(tf.float32, shape=[None, y_size])

	x_image = tf.reshape(x, [-1,28,28,1])

	keep_prob = tf.placeholder(tf.float32)

	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, y_size])
	b_fc2 = bias_variable([y_size])

	y = tf.matmul(h_fc1, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())

def pretrain_model((train_data_x, train_data_y), (test_data_x, test_data_y), batch_size=100, num_epochs=300, num_conv_layers=2, num_fc_layers=2):
	sess = tf.Session()
	x_size = train_data_x[0].size #reduce(lambda x, y: x*y, trainDataX[0].size)
	y_size = len(train_data_y[0])
	x = tf.placeholder(tf.float32, shape=[None, x_size])
	y_ = tf.placeholder(tf.float32, shape=[None, y_size])

	x_image = tf.reshape(x, [-1,28,28,1])

	keep_prob = tf.placeholder(tf.float32)

	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, y_size])
	b_fc2 = bias_variable([y_size])

	y = tf.matmul(h_fc1, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())
	train_size = len(train_data_y)
	accs = [0,1]
	epoch = -1
	while abs(accs[-1] - accs[-2]) > 1e-8:
		epoch+=1
		batch_num = epoch % train_size/batch_size
		start = batch_num*batch_size
		end = min(start + batch_size, train_size)
  		batch_x = train_data_x[start:end]
  		batch_y = train_data_y[start:end]
  		train_step.run(session=sess, feed_dict={x: batch_x, y_: batch_y})
		if epoch%100 == 0:
			train_accuracy = accuracy.eval(session=sess, feed_dict={
				x: test_data_x, y_: test_data_y})
			print("step %d, training accuracy %.4f"%(epoch, train_accuracy))
			accs.append(train_accuracy)
	print "finished pretraining at %i epochs." % epoch
	acc = accuracy.eval(session=sess, feed_dict={
	    x: test_data_x, y_: test_data_y})
	print("test accuracy %g"%acc)

	# print W_conv1.eval(session=sess)
	# print b_conv1.eval(session=sess)
	# print W_fc2.eval(session=sess)

	save_numpy(W_conv1.eval(session=sess), "W_conv1")
	save_numpy(b_conv1.eval(session=sess), "b_conv1")
	save_numpy(W_conv2.eval(session=sess), "W_conv2")
	save_numpy(b_conv2.eval(session=sess), "b_conv2")
	save_numpy(W_fc1.eval(session=sess), "W_fc1")
	save_numpy(b_fc1.eval(session=sess), "b_fc1")
	save_numpy(W_fc2.eval(session=sess), "W_fc2")
	save_numpy(b_fc2.eval(session=sess), "b_fc2")

def update_model(pretrain, data_source="mnist", num_conv_layers=2, num_fc_layers=2):
	sess = tf.Session()
	if data_source=="mnist":
		data = input_data.read_data_sets('MNIST_data', one_hot=True, )
		x_size = 784
		y_size = 10
	else:
		#TODO: update dataset
		data = input_data.read_data_sets('MNIST_data', one_hot=True)
	if pretrain:
		W_conv1 = tf.Variable(load_numpy("W_conv1"), name="W_conv1")
		b_conv1 = tf.Variable(load_numpy("b_conv1"), name="b_conv1")
		W_conv2 = tf.Variable(load_numpy("W_conv2"), name="W_conv2")
		b_conv2 = tf.Variable(load_numpy("b_conv2"), name="b_conv2")
		W_fc1 = tf.Variable(load_numpy("W_fc1"), name="W_fc1")
		b_fc1 = tf.Variable(load_numpy("b_fc1"), name="b_fc1")
		W_fc2 = tf.Variable(tf.random_normal([1024, y_size]), name="W_fc2")
		b_fc2 = tf.Variable(tf.random_normal([y_size]), name="b_fc2")
	else:
		W_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32]), name="W_conv1")
		b_conv1 = tf.Variable(tf.random_normal([32]), name="b_conv1")
		W_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64]), name="W_conv2")
		b_conv2 = tf.Variable(tf.random_normal([64]), name="b_conv2")
		W_fc1 = tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name="W_fc1")
		b_fc1 = tf.Variable(tf.random_normal([1024]), name="b_fc1")
		W_fc2 = tf.Variable(tf.random_normal([1024, y_size]), name="W_fc2")
		b_fc2 = tf.Variable(tf.random_normal([y_size]), name="b_fc2")

	x = tf.placeholder(tf.float32, shape=[None, x_size])
	y_ = tf.placeholder(tf.float32, shape=[None, y_size])

	x_image = tf.reshape(x, [-1,28,28,1])

	keep_prob = tf.placeholder(tf.float32)

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	y = tf.matmul(h_fc1, W_fc2) + b_fc2

	return (x, y, y_, sess, y_size, data)

def train_model((x, y, y_, sess, y_size, data), num_train=1, batch_size=100, num_epochs=1000):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())
	counts = [0]*y_size
	batches = []
	time = []
	accs = [0, 1]
	random_start = random.randint(0, len(data.train.labels))
	data.train.next_batch(random_start)
	while sum(counts) < y_size*num_train:
		batch = [[],[]]
		while len(batch[0]) < batch_size and sum(counts) < y_size*num_train:
			sample = data.train.next_batch(1)
			if counts[rev_one_hot(sample[1][0])] < num_train:
				batch[0].append(sample[0][0])
				batch[1].append(sample[1][0])
				counts[rev_one_hot(sample[1][0])]+=1
		batches.append(batch)
	num_batches = len(batches)
	i=-1
	while abs(accs[-1] - accs[-2]) > 1e-8:
		i+=1
		if num_train>0: batch = batches[i % num_batches]
		else: batch = data.train.next_batch(batch_size)
		if i%100 == 0:
			train_accuracy = accuracy.eval(session=sess, feed_dict={
				x: data.validation.images, y_: data.validation.labels})
			time.append(i)
			accs.append(train_accuracy)
			print("step %d, training accuracy %.4f"%(i, train_accuracy))
		train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1]})
	print "number of epochs until convergence:",i
	return (x, y, y_, sess, data, accuracy), (time, accs[2:])

def test_model((x, y, y_, sess, data, accuracy)):
	print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    	x: data.validation.images, y_: data.validation.labels}))


# plt.plot(time, errors)
# plt.show()
# plt.savefig("errors_"+str(train_accuracy)+".png")