import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path

class QDDataset(Dataset):
    def __init__(self, ds_path, phase):
        '''
        :param ds_path: path to directory that contains Annotations, train, val, test
        :param phase: train, val, or test
        '''
        self.ds_path = ds_path
        self.phase = phase
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # load dataset
        self.dataset = {}
        annotation_file = os.path.join('Annotations', f'{phase}.json')
        if annotation_file != None:
            print('loading dataset into memory...')
            dataset = json.load(open(annotation_file, 'r'))
            self.dataset = dataset
        if False: # Change to False if already have transformed data
            self.preprocess()
        print('Preprocessing done.')

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
        img_fpath = os.path.join(self.ds_path, 'Transformed', self.phase, image_id) + '.pt'
        input_tensor = torch.load(img_fpath)
        return input_tensor, torch.Tensor(flaws)

    def preprocess(self):
        os.makedirs(os.path.join(self.ds_path, 'Transformed', self.phase), exist_ok=True)
        for idx in tqdm(range(len(self.dataset))):
            # Get image and convert to tensor
            image_id = self.dataset[idx]['image']
            img_fpath = os.path.join(self.ds_path, 'Images', self.phase, image_id)
            tran_fpath = os.path.join(self.ds_path, 'Transformed', self.phase, image_id) + '.pt'
            input_image = Image.open(img_fpath)
            torch.save(self.transform(input_image), tran_fpath)