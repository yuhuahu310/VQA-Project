import torch
from torch.utils.data import DataLoader, Dataset
import os
import skimage.io as io
from skimage.transform import resize
from nltk.tokenize import RegexpTokenizer
import numpy as np
from vqa import VQA

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"

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
            self.vocab = np.concatenate(([SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, OOV_TOKEN], self.vqa.get_vocab()))
        else:
            self.vocab = np.concatenate(([SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, OOV_TOKEN], VQA(annotation_file=os.path.join(ds_path, 'Annotations/train.json')).get_vocab()))
        # import pdb;pdb.set_trace()
        self.reverse_vocab = dict(list(enumerate(self.vocab)))
        self.vocab = dict(zip(self.vocab, np.arange(len(self.vocab))))
        self.ds_path = ds_path
        self.phase = phase

    def __len__(self):
        return len(self.vqa.dataset)
    
    def _sent_2_idx_seq(self, sent):
        tokenizer = RegexpTokenizer(r'\w+')
        vec = [self.vocab[SOS_TOKEN]] + [self.vocab[word] if word in self.vocab else self.vocab[OOV_TOKEN] for word in tokenizer.tokenize(sent)] + [self.vocab[EOS_TOKEN]]
        return vec
    
    def idx_seq_2_sent(self, seq, to_str=True):
        if to_str:
            return ' '.join([self.reverse_vocab[idx] for idx in seq if self.reverse_vocab[idx] not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]])
        else:
            return [self.reverse_vocab[idx] for idx in seq]

    def __getitem__(self, idx):
        '''Return images, questions, and answers
        '''
        qa_pair = self.vqa.dataset[idx]
        image_id = qa_pair['image']
        answers = qa_pair['answers']
        question = qa_pair['question']

        # Get image and convert to tensor
        img_fpath = os.path.join(self.ds_path, "Images", self.phase, image_id)
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
    
class QADataset(VQADataset):
    def __init__(self, ds_path, phase, tokenize=True):
        super().__init__(ds_path, phase)
        self.tokenize = tokenize

    def __getitem__(self, idx):
        '''Return only questions and all answers as idx seqs
        '''
        qa_pair = self.vqa.dataset[idx]
        answers = qa_pair['answers']
        question = qa_pair['question']
        flat_answers = [i['answer'] for i in answers]
        # In case of a tie, pick the first. Might consider pick randomly
        answer = max(flat_answers, key=flat_answers.count) 

        if not self.tokenize:
            return question, answer
        
        q_vec = self._sent_2_idx_seq(question)
        a_vec = self._sent_2_idx_seq(answer)
        return torch.tensor(q_vec, dtype=torch.int), torch.tensor(a_vec, dtype=torch.int)


def collate_fn_pad(batch):
    questions = torch.nn.utils.rnn.pad_sequence([t[0] for t in batch], padding_value=2)
    answers = torch.nn.utils.rnn.pad_sequence([t[1] for t in batch], padding_value=2)
    return questions, answers

if __name__ == '__main__':
    vqa_dataset = VQADataset("../data", "train")
    print(f'len of dataset is {len(vqa_dataset)}')
    print(vqa_dataset.getQA(0))