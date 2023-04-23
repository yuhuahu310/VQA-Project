import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import json

class QDDataset(Dataset):
    def __init__(self, ds_path, phase):
        '''
        :param ds_path: path to quality_detector
        :param phase: train, val, or test
        '''
        self.ds_path = ds_path
        self.phase = phase
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # load dataset
        self.dataset = {}
        annotation_file = os.path.join(ds_path, 'Annotations', f'{phase}.json')
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
        input_image = Image.open(img_fpath)
        input_tensor = self.preprocess(input_image)
        return input_tensor, torch.Tensor(flaws)