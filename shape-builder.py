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

def generateRandomImages(folderPath, n, sizeX, sizeY):
	if not os.path.exists(folderPath):
		os.makedirs(folderPath)
	for i in range(n):
		print folderPath+"/img"+str(i)+".png"
		generateOneRandomImage(folderPath+"/img"+str(i)+".png", sizeX, sizeY)

def generateOneRandomImage(imgPath, sizeX, sizeY):
	txt = Image.new('L', (sizeX, sizeY), (255))
	d = ImageDraw.Draw(txt)
	i = random.randint(0,1)
	maxSize = min(sizeX, sizeY)
	if i:
		x = np.random.randint(sizeX*1/4,sizeX*3/4)
		y = np.random.randint(sizeY*1/4,sizeY*3/4)
		center = (x,y)
		sideLen = random.randint(maxSize*1/10, maxSize*1/2)
		rotation = random.randint(0, 89)
		lt = (center[0]-sideLen/2, center[1]-sideLen/2); lt = rotate(center, lt, rotation)
		rt = (center[0]-sideLen/2, center[1]+sideLen/2); rt = rotate(center, rt, rotation)
		lb = (center[0]+sideLen/2, center[1]+sideLen/2); lb = rotate(center, lb, rotation)
		rb = (center[0]+sideLen/2, center[1]-sideLen/2); rb = rotate(center, rb, rotation)
		d.polygon([lt, rt, lb, rb], fill = 'black')
	else:
		randParams = (random.randint(sizeX*1/4,sizeX*3/4), random.randint(sizeY*1/4,sizeY*3/4), random.randint(maxSize*1/10, maxSize*1/4))
		d.ellipse((randParams[0]-randParams[2], randParams[1]-randParams[2], randParams[0]+randParams[2], randParams[1]+randParams[2]), fill = "black")
	txt.save(imgPath)

if __name__ == "__main__":
	# python shape-builder.py folderPath n sizeX sizeY
	## folderPath: path to folder where imgs will be saved (does not need to exist)
	## n: number of imgs to generate
	## sizeX: number of pixels in width
	## sizeY: number of pixels in height
	generateRandomImages(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
	