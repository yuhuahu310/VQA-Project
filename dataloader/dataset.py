import torch
from torch.utils.data import DataLoader, Dataset
import os
import skimage.io as io
from skimage.transform import resize
from nltk.tokenize import RegexpTokenizer
from torchvision import transforms
import numpy as np
from vqa import VQA
import clip
from PIL import Image
from torch.nn import functional as F
import json
import pytesseract
import random
from nltk.corpus import words
from tqdm import tqdm

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"
OCR_TOKEN = "ocr"
CUSTOM_TOKENS = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, OOV_TOKEN, OCR_TOKEN]

# TODO: add padding and field to tackle sos/eos
class VQADataset(Dataset):
    def __init__(self, ds_path, phase, use_all_ans=True, additional_vocab=[]):
        '''

        :param ds_path: path to directory that contains Annotations, train, val, test
        :param phase: train, val, or test
        '''
        self.vqa = VQA(annotation_file=os.path.join(ds_path, 'Annotations', f'{phase}.json'), use_all_ans=use_all_ans)
        # Get vocabulary using training data
        if phase == 'train':
            self.vocab = list(self.vqa.get_vocab()) + additional_vocab
        else:
            self.vocab = list(VQA(annotation_file=os.path.join(ds_path, 'Annotations/train.json')).get_vocab()) + additional_vocab
        self.vocab = CUSTOM_TOKENS + sorted(list(np.unique(self.vocab)))
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = dict(list(enumerate(self.vocab)))
        self.vocab = dict(zip(self.vocab, np.arange(len(self.vocab))))
        self.ds_path = ds_path
        self.phase = phase


    def __len__(self):
        return len(self.vqa.dataset)
    
    def _sent_2_idx_seq(self, sent):
        tokenizer = RegexpTokenizer(r'\w+')
        vec = [self.vocab[SOS_TOKEN]] + \
                [self.vocab[word] if (word in self.vocab or word == OCR_TOKEN) else self.vocab[OOV_TOKEN] for word in tokenizer.tokenize(sent)] + \
                [self.vocab[EOS_TOKEN]]
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
        img_fpath = os.path.join(self.ds_path, self.phase, image_id)
        img_arr = io.imread(img_fpath)
        img_arr = resize(img_arr, (224, 224))
        img_tensor = torch.tensor(img_arr)
        img_tensor = torch.permute(img_tensor, (2, 0, 1))


        # Convert answers and questions to index sequence
        qa_pair = self.vqa.dataset[idx]
        answers = qa_pair['answers']
        question = qa_pair['question']
        flat_answers = [i['answer'] for i in answers]
        # In case of a tie, pick the first. Might consider pick randomly
        answer = max(flat_answers, key=flat_answers.count)

        q_vec = self._sent_2_idx_seq(question)
        a_vec = self._sent_2_idx_seq(answer)
        return img_tensor, torch.tensor(q_vec, dtype=torch.int), torch.tensor(a_vec, dtype=torch.int), image_id

class VQA_mm_Dataset(VQADataset):
    def __init__(self, ds_path, phase, include_q_vector=True, load_ocr=None, use_all_ans=True, subset=False, truncate_ocr=True, max_text_len=50):
        
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.include_q_vector = include_q_vector
        self.ans_types = {
            'unanswerable': torch.tensor(0),
            'yes/no': torch.tensor(1),
            'number': torch.tensor(2),
            'other': torch.tensor(3)
        }
        
        ocr_vocab = []
        self.ocr_path = load_ocr
        if load_ocr is not None:
            self.ocr_results, ocr_vocab = self._load_ocr_results(load_ocr)
        self.truncate_ocr = truncate_ocr

        self.max_text_len = max_text_len

        super().__init__(ds_path, phase, use_all_ans=use_all_ans, additional_vocab=ocr_vocab)

        if subset:
            N = int(0.1 * len(self.vqa.dataset))
            self.vqa.dataset = random.sample(self.vqa.dataset, N)
            
    # @staticmethod
    # def _remove_dupe(original_list):
    #     new_list = []
    #     seen = set()
    #     for item in original_list:
    #         if item not in seen:
    #             seen.add(item)
    #             new_list.append(item)
    #     return new_list

    @staticmethod
    def _preprocess_ocr_results(ocr_path):
        with open(ocr_path, 'r') as f:
            ocr_results = json.load(f) # {image_filename: List[ocr tokens]}
        f.close()
        import enchant
        d = enchant.Dict("en_US")
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        for image_id, ocr_words in tqdm(ocr_results.items()):
            ocr_results[image_id] = [word for word in ocr_words if word != '' and (len(word)>1 or word.isdigit()) and d.check(word) and word not in stop_words]
            # ocr_results[image_id] = VQA_mm_Dataset._remove_dupe(ocr_results[image_id])

        ocr_results_new = {image_id: words for image_id, words in ocr_results.items() if len(words)>0}

        ocr_path_stem = ocr_path.replace(".json", "")
        with open(f"{ocr_path_stem}_words_only_no_dup.json", 'w') as f:
            json.dump(ocr_results_new, f, indent=2)
        f.close()
        return ocr_results_new

    @staticmethod
    def _load_ocr_results(ocr_path):
        additional_vocab = set()
        with open(ocr_path, 'r') as f:
            ocr_results = json.load(f) # {image_filename: List[ocr tokens]}
        f.close()
        for _, ocr_words in ocr_results.items():
            additional_vocab.update(ocr_words)
        return ocr_results, list(additional_vocab)


    def __getitem__(self, idx):
        '''Return images, questions, and answers
        '''
        qa_pair = self.vqa.dataset[idx]
        image_id = qa_pair['image']
        answers = qa_pair['answers']
        question = qa_pair['question']
        ans_type = self.ans_types[qa_pair['answer_type']]

        # Get image and convert to tensor
        img_fpath = os.path.join(self.ds_path, self.phase, image_id)
        # img_arr = io.imread(img_fpath)
        image = Image.open(img_fpath).convert("RGB")
        img_tensor = self.preprocess(image)
        
        ocr = ""
        if self.ocr_path is not None and image_id in self.ocr_results:
            ocr = self.ocr_results[image_id]
            if self.truncate_ocr:
                q_len = clip.tokenize(question)[0].count_nonzero().item()
                # print(clip.tokenize(question))
                max_ocr_len = max(self.max_text_len - q_len - 1, 0)
                # if max_ocr_len < len(ocr): print(max_ocr_len, len(ocr))
                ocr = ocr[:max_ocr_len]
            if ocr == "":
                ocr = ' '.join(ocr)
            else:
                ocr = ' ' + OCR_TOKEN + ' ' + ' '.join(ocr)
        # <sos> questions <ocr> ocr texts <eos>

        # Convert answers and questions to index sequence
        flat_answers = [i['answer'] for i in answers]
        # In case of a tie, pick the first. Might consider pick randomly
        answer = max(flat_answers, key=flat_answers.count)
        a_vec = self._sent_2_idx_seq(answer)

        question = clip.tokenize(question + ocr).squeeze(0)
        answer = clip.tokenize(answer).squeeze(0)
        # ocr_tokenized = clip.tokenize(ocr).squeeze(0)
        if self.include_q_vector:
            q_vec = self._sent_2_idx_seq(qa_pair['question'] + ocr)
            # ocr_vec = self._sent_2_idx_seq(ocr) 
            return img_tensor, question, answer, torch.tensor(a_vec, dtype=torch.int), image_id, torch.tensor(q_vec, dtype=torch.int), ans_type

        return img_tensor, question, answer, torch.tensor(a_vec, dtype=torch.int), image_id, ans_type

class VQA_mm2_Dataset(VQADataset):
    def __init__(self, ds_path, phase, img_transforms=None, tokenizer=None, include_q_vector=True, load_ocr=None):
        super().__init__(ds_path, phase)
        self.img_transforms = img_transforms
        self.tokenizer = tokenizer
        self.include_q_vector = include_q_vector
        self.load_ocr = load_ocr

    def __getitem__(self, idx):
        '''Return images, questions, and answers
        '''
        qa_pair = self.vqa.dataset[idx]
        image_id = qa_pair['image']
        answers = qa_pair['answers']
        question = qa_pair['question']

        # Get image and convert to tensor
        img_fpath = os.path.join(self.ds_path, self.phase, image_id)
        image = Image.open(img_fpath).convert("RGB")
        # OCR: online/offline
        img_tensor = self.img_transforms(image)
        if self.load_ocr:
            with open(self.load_ocr) as ocr_dict:
                ocr = json.load(ocr_dict)[str(image_id)]
        else:
            ocr = pytesseract.image_to_string(image)
        ocr = OCR_TOKEN + ocr

        # Convert answers and questions to index sequence
        flat_answers = [i['answer'] for i in answers]
        # In case of a tie, pick the first. Might consider pick randomly
        answer = max(flat_answers, key=flat_answers.count)
        a_vec = self._sent_2_idx_seq(answer)
        question = self.tokenizer.encode_plus(question, add_special_tokens=True, return_tensors = 'pt')['input_ids'][0]
        ocr_tokenized = self.tokenizer.encode_plus(ocr, add_special_tokens=True, return_tensors = 'pt')['input_ids'][0]
        if self.include_q_vector:
            q_vec = self._sent_2_idx_seq(qa_pair['question'])
            ocr_vec = self._sent_2_idx_seq(ocr)
            return img_tensor, question + ocr_tokenized, torch.tensor(a_vec, dtype=torch.int), image_id, torch.tensor(q_vec + ocr_vec, dtype=torch.int)

        return img_tensor, question + ocr_tokenized,  torch.tensor(a_vec, dtype=torch.int), image_id

class QADataset(VQADataset):
    def __init__(self, ds_path, phase, tokenize=True, include_imageid=False):
        super().__init__(ds_path, phase)
        self.tokenize = tokenize
        self.include_imageid = include_imageid

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
            if self.include_imageid:
                return question, answer, qa_pair['image']
            else:
                return question, answer
        
        q_vec = self._sent_2_idx_seq(question)
        a_vec = self._sent_2_idx_seq(answer)
        if self.include_imageid:
            return torch.tensor(q_vec, dtype=torch.int), torch.tensor(a_vec, dtype=torch.int), qa_pair['image']
        else:
            return torch.tensor(q_vec, dtype=torch.int), torch.tensor(a_vec, dtype=torch.int)


def collate_fn_pad(batch):
    questions = torch.nn.utils.rnn.pad_sequence([t[0] for t in batch], padding_value=2)
    answers = torch.nn.utils.rnn.pad_sequence([t[1] for t in batch], padding_value=2)
    image_ids = [t[2] for t in batch]
    return questions, answers, image_ids


def collate_fn_pad_image(batch):
    questions = torch.nn.utils.rnn.pad_sequence([t[1] for t in batch], padding_value=2, batch_first=True)
    answers = torch.nn.utils.rnn.pad_sequence([t[2] for t in batch], padding_value=2, batch_first=True)
    img_list = [t[0] for t in batch]
    images = torch.stack(img_list, dim=0)
    image_ids = [t[3] for t in batch]
    return images, questions, answers, image_ids

# batch = (img_tensor, question + ocr_tokenized, answer, torch.tensor(a_vec, dtype=torch.int), image_id, torch.tensor(q_vec + ocr_vec, dtype=torch.int), ans_type)
def collate_fn_pad_mm(batch):
    img_list = [t[0] for t in batch]
    images = torch.stack(img_list, dim=0)
    image_ids = [t[4] for t in batch]

    anstype_list = [t[-1] for t in batch]
    anstypes = torch.tensor(anstype_list)
    
    # clip tokenized
    question_list = [t[1] for t in batch]
    questions = torch.stack(question_list, dim=0)

    answers = torch.nn.utils.rnn.pad_sequence([t[3] for t in batch], padding_value=2, batch_first=True)
    questions_vec = torch.nn.utils.rnn.pad_sequence([t[5] for t in batch], padding_value=2, batch_first=True)
    return images, questions, answers, anstypes, image_ids, questions_vec

# img_tensor, question, answer, torch.tensor(a_vec, dtype=torch.int), image_id
def collate_fn_pad_mm2(batch):
    img_list = [t[0] for t in batch]
    images = torch.stack(img_list, dim=0)
    image_ids = [t[3] for t in batch]
    
    # tokenized
    questions = torch.nn.utils.rnn.pad_sequence([t[1] for t in batch], padding_value=0, batch_first=True)
    answers_vec = torch.nn.utils.rnn.pad_sequence([t[2] for t in batch], padding_value=2, batch_first=True)
    questions_vec = torch.nn.utils.rnn.pad_sequence([t[4] for t in batch], padding_value=2, batch_first=True)
    return images, questions, answers_vec, questions_vec, image_ids


if __name__ == '__main__':
    # for phase in ['train', 'val']:
    #     path = f'ocr_results/ocr_texts_{phase}.json'
    #     VQA_mm_Dataset._preprocess_ocr_results(path)

    vqa_dataset = VQA_mm_Dataset("../data", "train", load_ocr='ocr_results/ocr_texts_train_words_only.json')
    # vqa_dataset = VQA_mm_Dataset("../data", "train")
    # # breakpoint()
    print(f'len of dataset is {len(vqa_dataset)}')
    print(vqa_dataset[0])