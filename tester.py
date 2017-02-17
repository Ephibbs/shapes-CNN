import sys
from cnn import *
import shape_builder
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy
import glob
from PIL import Image, ImageDraw, ImageFont

def img2array(path):
    img = Image.open(path).convert('F')
    return numpy.array(img).ravel()

def one_hotify(arr):
	oh_arr = []
	m = max(arr)
	for a in arr:
		el = [0]*(m+1)
		el[a] = 1
		oh_arr.append(el)
	return oh_arr

def file2array(f):
	arr = []
	for line in f:
		arr.append(int(line.split("\n")[0]))
	return arr

if __name__ == "__main__":
	# python tester.py "test-suite"|"single-test" (0 (no pretraining)|1 (pretrain)| 2 (use saved model)) (number of training examples per class?)
	if len(sys.argv) < 3: 
		print 'Usage: python tester.py "test-suite"|"single-test" [ 0 (no pretraining) | 1 (pretrain)| 2 (use saved pretrained model) ] [# training samples per class]'
		print 'Running python tester.py "test-suite" does not require any more parameters'
	mode = sys.argv[1]
	batch_size = 50
	if mode=="test-suite":
		image_path = "shape-imgs"
		filenames = glob.glob(image_path+"_nparrays"+"/*.npy")
		data_x = numpy.array([numpy.load(fn).ravel() for fn in filenames])
		f = open(image_path+"_nparrays"+"/labels.txt", "r")
		data_y = one_hotify(file2array(f))
		f.close()
		for num_train in [1, 10, 100, 1000, 0]:
			for pretrain in [0,2]:
				for i in range(3):
					if pretrain == 1: 
						pretrain_model((data_x, data_y), (data_x, data_y), batch_size=batch_size, num_conv_layers=2, num_fc_layers=2)
					else: 
						create_model((data_x, data_y), (data_x, data_y), num_conv_layers=2, num_fc_layers=2)
					params = update_model(pretrain, "mnist", num_conv_layers=2, num_fc_layers=2)
					params, train_accs = train_model(params, num_train=num_train, batch_size=batch_size)
					print "with numtrain=%i, pretrain=%i, and test #%i\n" % (num_train, pretrain, i)
					test_model(params)
					print "\n\n"
	elif mode=="single-test":
		pretrain = int(sys.argv[2])
		num_train = int(sys.argv[3])
		if pretrain:
			print "Setting up a model WITH pretraining, training on %i samples per class ..." % num_train
		else:
			print "Setting up a model WITHOUT pretraining, training on %i samples per class ..." % num_train
		image_path = "shape-imgs"
		filenames = glob.glob(image_path+"_nparrays"+"/*.npy")
		data_x = numpy.array([numpy.load(fn).ravel() for fn in filenames])
		f = open(image_path+"_nparrays"+"/labels.txt", "r")
		data_y = one_hotify(file2array(f))
		f.close()
		if pretrain == 1: 
			pretrain_model((data_x, data_y), (data_x, data_y), batch_size=batch_size, num_conv_layers=2, num_fc_layers=2)
		else: 
			create_model((data_x, data_y), (data_x, data_y), num_conv_layers=2, num_fc_layers=2)
		params = update_model(pretrain, "mnist", num_conv_layers=2, num_fc_layers=2)
		params, train_accs = train_model(params, num_train=num_train, batch_size=batch_size)
		test_model(params)