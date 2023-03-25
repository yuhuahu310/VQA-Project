import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import skimage.io as io
from skimage.transform import resize
from nltk.tokenize import RegexpTokenizer
import numpy as np
from vqa import VQA


class VQADataset(Dataset):
    def __init__(self, ds_path, phase):
        '''

        :param ds_path: path to directory that contains Annotations, train, val, test
        :param phase: train, val, or test
        '''
        self.vqa = VQA(annotation_file=os.path.join(ds_path, 'Annotations', f'{phase}.json'))
        # Get vocabulary using training data
        if phase == 'train':
            self.vocab = self.vqa.get_vocab()
        else:
            self.vocab = VQA(annotation_file=os.path.join(ds_path, 'Annotations/train.json')).get_vocab()
        self.ds_path = ds_path
        self.phase = phase

    def __len__(self):
        return len(self.vqa.dataset)

    def __getitem__(self, idx):
        qa_pair = self.vqa.dataset[idx]
        image_id = qa_pair['image']
        answers = qa_pair['answers']
        question = qa_pair['question']

        # Get image and convert to tensor
        img_fpath = os.path.join(self.ds_path, self.phase, image_id)
        img_arr = io.imread(img_fpath)
        img_arr = resize(img_arr, (800, 600))
        img_tensor = torch.tensor(img_arr)

        # TODO: Convert answers and questions to one-hot vectors
        answer_vocab = self.vqa.get_answers_vocab()
        answer_vec = torch.zeros(len(answer_vocab))
        for ans in answers:
            ans_idx = answer_vocab[ans['answer']]
            answer_vec[ans_idx] = 1


        return img_tensor

