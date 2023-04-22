import torch
from torch.utils.data import DataLoader, Dataset
import os
import skimage.io as io
from skimage.transform import resize
from torchvision import transforms
import numpy as np
from vqa import VQA
from PIL import Image
from torch.nn import functional as F
import json

class QDDataset(Dataset):
    def __init__(self, ds_path, phase):
        '''
        :param ds_path: path to directory that contains quality_detector
        :param phase: train, val, or test
        '''
        self.ds_path = ds_path
        self.phase = phase
        # load dataset
        self.dataset = {}
        annotation_file = os.path.join(ds_path, 'quality_detector/Annotations', f'{phase}.json')
        if annotation_file != None:
            print('loading dataset into memory...')
            dataset = json.load(open(annotation_file, 'r'))
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        '''Return images and vector of flaws
        '''
        image_id = self.dataset[idx]['image']
        flaws_dict = self.dataset[idx]['flaws']
        flaws = []
        for k in sorted(flaws_dict):
            flaws.append(flaws_dict[k])
        

        # Get image and convert to tensor
        img_fpath = os.path.join(self.ds_path, self.phase, image_id)
        img_arr = io.imread(img_fpath)
        img_arr = resize(img_arr, (224, 224))
        img_tensor = torch.tensor(img_arr)
        img_tensor = torch.permute(img_tensor, (2, 0, 1))