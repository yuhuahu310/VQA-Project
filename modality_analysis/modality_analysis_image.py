import sys
sys.path.insert(0, '../dataloader')
from vqa import VQA
import skimage.io as io
from skimage.transform import resize
import os
import matplotlib.pyplot as plt
import numpy as np

dataDir='../data'
split = 'val'
annFile='%s/Annotations/%s.json'%(dataDir, split)
imgDir = '%s/Images/%s/' %(dataDir, split)
savefigDir = 'fig/'

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
    # statistics of low-quality images
    unsuitable_cnt = [num_unsuitable(ann) for ann in anns]
    filtered = [cnt for cnt in unsuitable_cnt if cnt > 0]
    print("Percentage of images that are considered unsuitable by at least 1 annotators:", len(filtered) / len(unsuitable_cnt)*100)
    
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
        avg_arr += img_arr
        cnt += 1
    avg_arr /= N
    io.imsave(f"{savefigDir}/avg_img.png", avg_arr)


analysis_images()