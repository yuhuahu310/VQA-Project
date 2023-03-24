from ...API.PythonHelper.vqaTools.vqa import VQA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset



#
# # Load dataset
# vqa = VQA(annotation_file='v2_mscoco_train2014_annotations.json')
# anns = vqa.getAnns(ansTypes=['yes/no'])
# questions = [ann['question'] for ann in anns]
# answers = [ann['answers'][0]['answer'] for ann in anns]  # TODO: deal with multiple answers
# data = list(zip(questions, answers))
# train_data, val_data = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]
#
# # Tokenize text
# from nltk.tokenize import word_tokenize
# from collections import Counter
#
# words = [word_tokenize(q.lower()) for q, a in train_data]
# word_freq = Counter([w for q in words for w in q])
# vocab = sorted(word_freq, key=word_freq.get, reverse=True)
# vocab_to_int = {word: i + 1 for i, word in enumerate(vocab)}
# train_tokens = [[vocab_to_int[w] for w in q] for q in words]
# train_labels = [1 if a == 'yes' else 0 for q, a in train_data]


# Create PyTorch DataLoader
class VQADataset(Dataset):
    def __init__(self, vqa):
        self.vqa = vqa
        self.vocab = self.vqa.get_vocab()


    def __len__(self):
        return len(self.vqa.dataset)

    def __getitem__(self, idx):
        qa_pair = self.vqa.dataset[idx]
        image_id = qa_pair['image']
        answers = qa_pair['answers']

        # TODO: preprocess question and answers


        # Get image and convert to tensor
        image_tensor = torch.tensor()

        # Convert answers to one-hot vectors
        answer_vocab = self.vqa.get_answers_vocab()
        answer_vec = torch.zeros(len(answer_vocab))
        for ans in answers:
            ans_idx = answer_vocab[ans['answer']]
            answer_vec[ans_idx] = 1

        return image_tensor, answer_vec


# Instantiate VQA class and VQADataset
vqa = VQA(annotation_file='../../Annotations/val.json')
vqa_dataset = VQADataset(vqa)
print(len(vqa_dataset))


#
# # Instantiate DataLoader
# batch_size = 32
# vqa_dataloader = DataLoader(vqa_dataset, batch_size=batch_size, shuffle=True)