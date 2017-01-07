import numpy as np
import math
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import random

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def generateRandomImages(folderPath, n, sizeX, sizeY, labelFile="labels.txt"):
	f = open(labelFile, "w+")
	if not os.path.exists(folderPath):
		os.makedirs(folderPath)
	if not os.path.exists(folderPath+"_nparrays"):
		os.makedirs(folderPath+"_nparrays")
	for i in range(n):
		print folderPath+"/img"+str(i)+".png"
		label = generateOneRandomImage(folderPath, "img"+str(i), sizeX, sizeY)
		f.write("%i\n" % label)
	f.close()

def generateOneRandomImage(folderPath, imgPath, sizeX, sizeY, antialias=2):
	#times antialias for anti-aliasing
	largeSizeX = antialias*sizeX
	largeSizeY = antialias*sizeY
	txt = Image.new('L', (largeSizeX, largeSizeY), (0))
	d = ImageDraw.Draw(txt)
	i = random.randint(0,1)
	maxSize = min(largeSizeX, largeSizeY)
	if i:
		x = np.random.randint(largeSizeX*1/4,largeSizeX*3/4)
		y = np.random.randint(largeSizeY*1/4,largeSizeY*3/4)
		center = (x,y)
		sideLen = random.randint(maxSize*1/10, maxSize*1/2)
		rotation = random.randint(0, 89)
		lt = (center[0]-sideLen/2, center[1]-sideLen/2); lt = rotate(center, lt, rotation)
		rt = (center[0]-sideLen/2, center[1]+sideLen/2); rt = rotate(center, rt, rotation)
		lb = (center[0]+sideLen/2, center[1]+sideLen/2); lb = rotate(center, lb, rotation)
		rb = (center[0]+sideLen/2, center[1]-sideLen/2); rb = rotate(center, rb, rotation)
		d.polygon([lt, rt, lb, rb], fill = 'white')
	else:
		randParams = (random.randint(largeSizeX*1/4,largeSizeX*3/4), random.randint(largeSizeY*1/4,largeSizeY*3/4), random.randint(maxSize*1/10, maxSize*1/4))
		d.ellipse((randParams[0]-randParams[2], randParams[1]-randParams[2], randParams[0]+randParams[2], randParams[1]+randParams[2]), fill = "white")
	txt.thumbnail((sizeX, sizeY), Image.ANTIALIAS)
	narr = np.array(txt).astype(np.float32)/255
	#print narr
	np.save(folderPath+"_nparrays"+"/"+imgPath+".npy", narr)
	txt.save(folderPath+"/"+imgPath+".png")
	return i

if __name__ == "__main__":
	# python shape-builder.py folderPath n sizeX sizeY
	## folderPath: path to folder where imgs will be saved (does not need to exist)
	## n: number of imgs to generate
	## sizeX: number of pixels in width
	## sizeY: number of pixels in height
	generateRandomImages(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
	