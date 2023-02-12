from vqaTools.vqa import VQA
import random
import skimage.io as io
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import nltk
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter
import numpy as np

dataDir='../../'
split = 'val'
annFile='%s/Annotations/%s.json'%(dataDir, split)
imgDir = '%s/Images/%s/' %(dataDir, split)
savefigDir = '%s/Figs/' %(dataDir)	

print(f"Annotations are at {annFile}")
print(f"Images are at {imgDir}")

vqa=VQA(annFile)
# load and display QA annotations for given answer types
"""	
ansTypes can be one of the following
yes/no
number
other
unanswerable
"""
anns = vqa.getAnns()

def num_unsuitable(anno):
    count = 0
    for ans in anno["answers"]:
        ans_text = ans["answer"]
        if "unsuitable" in ans_text:
            count += 1
    return count

def analysis_images():
    # unsuitable_cnt = [num_unsuitable(ann) for ann in anns]
    # filtered = [cnt for cnt in unsuitable_cnt if cnt > 0]
    # print("Percentage of images that are considered unsuitable by at least 1 annotators:", len(filtered) / len(unsuitable_cnt)*100)
    
    # image diversity
    imlist = [os.path.join(imgDir, anno["image"]) for anno in anns]
    img_shape = (800, 600, 3)
    N = len(imlist)
    avg_arr = np.zeros(img_shape)
    cnt = 1
    for img in imlist:
        print(f"{cnt} out of {N}")
        img_arr = io.imread(img)
        img_arr = resize(img_arr, (img_shape[0], img_shape[1]))
        print(img_arr)
        avg_arr += img_arr
        cnt += 1
        if cnt == 6:
            break
    avg_arr /= N
    print(avg_arr)
    io.imsave(f"{savefigDir}/avg_img_test.png", avg_arr)

# def analysis_answers(): 
#     answers = [ann['answers'] for ann in anns]
#     print(answers)
#     pass

analysis_images()