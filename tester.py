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
	# python tester.py (pretain? 0:use new nonpretrained model; 1:use new pretrained model; 2:use saved pretrained model ) (number of training examples per class?) (Pretrain epochs) (training epochs) (batchSize)
	pretrain = int(sys.argv[1])
	numTrain = int(sys.argv[2])
	numPreEpochs = int(sys.argv[3])
	numEpochs = int(sys.argv[4])
	batchSize = int(sys.argv[5])
	imagePath = "shape-imgs"
	filenames = glob.glob(imagePath+"_nparrays"+"/*")
	dataX = numpy.array([numpy.load(fn).ravel() for fn in filenames])
	f = open("labels.txt", "r")
	dataY = one_hotify(file2array(f))
	f.close()
	if pretrain == 1: 
		pretrainAndSaveModel((dataX, dataY), (dataX, dataY), batchSize=batchSize, numEpochs=numPreEpochs, numConvLayers=2, numFCLayers=2)
	else: 
		createModel((dataX, dataY), (dataX, dataY), numConvLayers=2, numFCLayers=2)
	trainAndTestModel(pretrain, "mnist", numTrain=numTrain, batchSize=batchSize, numEpochs=numEpochs, numConvLayers=2, numFCLayers=2)