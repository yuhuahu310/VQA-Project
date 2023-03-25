import torch
from torch.utils.data import DataLoader, Dataset
import os
import skimage.io as io
from skimage.transform import resize
from nltk.tokenize import RegexpTokenizer
import numpy as np
from vqa import VQA

# TODO: add padding and field to tackle sos/eos
class VQADataset(Dataset):
    def __init__(self, ds_path, phase):
        '''

        :param ds_path: path to directory that contains Annotations, train, val, test
        :param phase: train, val, or test
        '''
        self.vqa = VQA(annotation_file=os.path.join(ds_path, 'Annotations', f'{phase}.json'))
        # Get vocabulary using training data
        if phase == 'train':
            self.vocab = ["<sos>", "<eos>"] + self.vqa.get_vocab()
        else:
            self.vocab = ["<sos>", "<eos>"] + VQA(annotation_file=os.path.join(ds_path, 'Annotations/train.json')).get_vocab()
        # import pdb;pdb.set_trace()
        self.vocab = dict(zip(self.vocab, np.arange(len(self.vocab))))
        self.ds_path = ds_path
        self.phase = phase

    def __len__(self):
        return len(self.vqa.dataset)
    
    def _sent_2_idx_seq(self, sent):
        tokenizer = RegexpTokenizer(r'\w+')
        vec = ['<sos>'] + [self.vocab[word] for word in tokenizer.tokenize(sent)] + ['<eos>']
        return vec

    def getQA(self, idx):
        '''Return only questions and all answers as idx seqs
        '''
        qa_pair = self.vqa.dataset[idx]
        answers = qa_pair['answers']
        question = qa_pair['question']
        q_vec = self._sent_2_idx_seq(question)

        flat_answers = [i['answer'] for i in answers]
        a_vec = [self._sent_2_idx_seq(ans) for ans in flat_answers]
        return q_vec, a_vec

    def __getitem__(self, idx):
        '''Return images, questions, and answers
        '''
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
        # answer_vocab = self.vqa.get_answers_vocab()
        # answer_vec = torch.zeros(len(answer_vocab))
        # for ans in answers:
        #     ans_idx = answer_vocab[ans['answer']]
        #     answer_vec[ans_idx] = 1



        return img_tensor

if __name__ == '__main__':
    vqa_dataset = VQADataset("../data", "train")
    print(f'len of dataset is {len(vqa_dataset)}')
    print(vqa_dataset.getQA(0))