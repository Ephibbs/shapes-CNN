import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import numpy

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

def createModel((trainDataX, trainDataY), (testDataX, testDataY), numConvLayers=2, numFCLayers=2):
	sess = tf.InteractiveSession()
	xSize = trainDataX[0].size #reduce(lambda x, y: x*y, trainDataX[0].size)
	ySize = len(trainDataY[0])
	x = tf.placeholder(tf.float32, shape=[None, xSize])
	y_ = tf.placeholder(tf.float32, shape=[None, ySize])

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

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, ySize])
	b_fc2 = bias_variable([ySize])

	y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()
	save_path = saver.save(sess, "unpretrained_models/model-%i-%i.ckpt"%(numConvLayers, numFCLayers))
	return save_path

def pretrainAndSaveModel((trainDataX, trainDataY), (testDataX, testDataY), batchSize=100, numEpochs=300, numConvLayers=2, numFCLayers=2):
	sess = tf.InteractiveSession()
	xSize = trainDataX[0].size #reduce(lambda x, y: x*y, trainDataX[0].size)
	ySize = len(trainDataY[0])
	x = tf.placeholder(tf.float32, shape=[None, xSize])
	y_ = tf.placeholder(tf.float32, shape=[None, ySize])

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

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, ySize])
	b_fc2 = bias_variable([ySize])

	y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())
	trainSize = len(trainDataY)
	for epoch in range(numEpochs):
		for batchNum in range(trainSize/batchSize):
			start = batchNum*batchSize
			end = min(start + batchSize, trainSize)
	  		batchX = trainDataX[start:end]
	  		batchY = trainDataY[start:end]
	  		train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 0.5})
		if epoch%1 == 0:
			train_accuracy = accuracy.eval(feed_dict={
				x: batchX, y_: batchY, keep_prob: 1.0})
			print("step %d, training accuracy %.4f"%(epoch, train_accuracy))
	acc = accuracy.eval(feed_dict={
	    x: testDataX, y_: testDataY, keep_prob: 1.0})
	print("test accuracy %g"%acc)

	saver = tf.train.Saver()
	save_path = saver.save(sess, "pretrained_models/model-%i-%i-%i-%ipct.ckpt"%(batchSize, numEpochs, trainSize, acc))
	return save_path

def trainAndTestModel(modelPath, dataSource="mnist", numTrain=1, batchSize=100, numEpochs=300, numConvLayers=2, numFCLayers=2):
	if dataSource=="mnist":
		data = input_data.read_data_sets('MNIST_data', one_hot=True)
		xSize = 784
		ySize = 10
	else:
		#TODO: update dataset
		data = input_data.read_data_sets('MNIST_data', one_hot=True)

	W_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32]), name="W_conv1")
	b_conv1 = tf.Variable(tf.random_normal([32]), name="b_conv1")
	W_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64]), name="W_conv2")
	b_conv2 = tf.Variable(tf.random_normal([64]), name="b_conv2")
	W_fc1 = tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name="W_fc1")
	b_fc1 = tf.Variable(tf.random_normal([1024]), name="b_fc1")
	W_fc2 = tf.Variable(tf.random_normal([1024, ySize]), name="W_fc2")
	b_fc2 = tf.Variable(tf.random_normal([ySize]), name="b_fc2")

	print "Restoring model from %s" % modelPath
	sess = tf.InteractiveSession()
	saver = tf.train.import_meta_graph(modelPath+".meta")
	#saver.restore(sess, tf.train.latest_checkpoint('./'))
	saver.restore(sess, modelPath)
	print("Model restored.")

	x = tf.placeholder(tf.float32, shape=[None, xSize])
	y_ = tf.placeholder(tf.float32, shape=[None, ySize])

	x_image = tf.reshape(x, [-1,28,28,1])

	keep_prob = tf.placeholder(tf.float32)

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())

	counts = [0]*ySize
	batches = []
	while sum(counts) < ySize*numTrain:
		batch = [[],[]]
		while len(batch[0]) < batchSize and sum(counts) < ySize*numTrain:
			sample = data.train.next_batch(1)
			print counts
			if counts[rev_one_hot(sample[1][0])] < numTrain:
				batch[0].append(sample[0][0])
				batch[1].append(sample[1][0])
				counts[rev_one_hot(sample[1][0])]+=1
		batches.append(batch)
	numBatches = len(batches)
	for i in range(numEpochs):
		if numTrain>0: batch = batches[i % numBatches]
		else: batch = data.train.next_batch(batchSize)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={
				x:batch[0], y_: batch[1], keep_prob: 1.0})
			print("step %d, training accuracy %.4f"%(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	print("test accuracy %g"%accuracy.eval(feed_dict={
    	x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

	# saver = tf.train.Saver()
	# with tf.Session() as sess:
 #  		sess.run(init_op)
	# 	save_path = saver.save(sess, "/trained_models/model-%i-%i-%i-%ipct.ckpt"%(batchSize, numEpochs, trainSize, acc))

