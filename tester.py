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
	# python tester.py (pretain?) (number of training examples per class?) (preimage directory path)
	pretrain = int(sys.argv[1])
	numTrain = int(sys.argv[2])
	imagePath = "shape-imgs" #sys.argv[3]
	filenames = glob.glob(imagePath+"_nparrays"+"/*")
	dataX = numpy.array([numpy.load(fn).ravel() for fn in filenames])
	f = open("labels.txt", "r")
	dataY = one_hotify(file2array(f))
	f.close()
	if pretrain: model_path = pretrainAndSaveModel((dataX, dataY), (dataX, dataY), batchSize=100, numEpochs=150, numConvLayers=2, numFCLayers=2)
	else: model_path = createModel((dataX, dataY), (dataX, dataY), numConvLayers=2, numFCLayers=2)
	trainAndTestModel(model_path, "mnist", numTrain=1, batchSize=100, numEpochs=500, numConvLayers=2, numFCLayers=2)