# coding: utf-8

from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

dataDir='../../'
split = 'val'
annFile='%s/Annotations/%s.json'%(dataDir, split)
imgDir = '%s/Images/%s/' %(dataDir, split)	

print(f"Annotations are at {annFile}")
print(f"Images are at {imgDir}")

# initialize VQA api for QA annotations
vqa=VQA(annFile)		
# load and display QA annotations for given answer types
"""	
ansTypes can be one of the following
yes/no
number
other
unanswerable
"""
anns = vqa.getAnns(ansTypes='yes/no');	   
randomAnn = random.choice(anns)	
vqa.showQA([randomAnn])
imgFilename = randomAnn['image']
print(imgFilename)
if os.path.isfile(imgDir + imgFilename):
	I = io.imread(imgDir + imgFilename)
	plt.imshow(I)
	plt.axis('off')
	plt.show()

# load and display QA annotations for given images
imgs = vqa.getImgs()
anns = vqa.getAnns(imgs=imgs)
randomAnn = random.choice(anns)
vqa.showQA([randomAnn])  
imgFilename = randomAnn['image']
if os.path.isfile(imgDir + imgFilename):
	I = io.imread(imgDir + imgFilename)
	plt.imshow(I)
	plt.axis('off')
	plt.show()
