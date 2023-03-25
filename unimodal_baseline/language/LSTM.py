from ...vqaTools.vqa import VQA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# class LSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
#         super().__init__()
        
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, text):
#         embedded = self.dropout(self.embedding(text))
#         output, (hidden, cell) = self.lstm(embedded)
#         hidden = self.dropout(hidden[-1,:,:])
#         return self.fc(hidden)

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the encoder class
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input):
        _, (hidden, cell) = self.lstm(input)
        return hidden, cell

# Define the decoder class
class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = self.linear(output.squeeze(0))
        return output, hidden, cell

# Define the Seq2Seq class that combines the encoder and decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, target):
        encoder_hidden, encoder_cell = self.encoder(input)

        decoder_input = torch.zeros((1, 1, self.decoder.input_size)).to(device)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        loss = 0
        for i in range(target.shape[0]):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            loss += nn.functional.cross_entropy(decoder_output, target[i])
            decoder_input = target[i].unsqueeze(0).unsqueeze(0)

        return loss


# Hyperparameters
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 2 # binary classification
N_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
N_EPOCHS = 10

# Load dataset
vqa = VQA(annotation_file='v2_mscoco_train2014_annotations.json')
anns = vqa.getAnns(ansTypes=['yes/no'])
questions = [ann['question'] for ann in anns]
answers = [ann['answers'][0]['answer'] for ann in anns] # TODO: deal with multiple answers
data = list(zip(questions, answers))
train_data, val_data = data[:int(0.8*len(data))], data[int(0.8*len(data)):]

# Tokenize text
from nltk.tokenize import word_tokenize
from collections import Counter

words = [word_tokenize(q.lower()) for q, a in train_data]
word_freq = Counter([w for q in words for w in q])
vocab = sorted(word_freq, key=word_freq.get, reverse=True)
vocab_to_int = {word: i+1 for i, word in enumerate(vocab)}
train_tokens = [[vocab_to_int[w] for w in q] for q in words]
train_labels = [1 if a == 'yes' else 0 for q, a in train_data]

# Create PyTorch DataLoader
class VQADataset(Dataset):
    def __init__(self, vqa):
        self.vqa = vqa
        
    def __len__(self):
        return len(self.vqa.dataset)
    
    def __getitem__(self, idx):
        qa_pair = self.vqa.dataset[idx]
        question = qa_pair['question']
        image_id = qa_pair['image']
        answers = qa_pair['answers']
        
        # TODO: preprocess question and answers
        
        # Convert question and image ID to tensors
        question_tensor = torch.tensor(question_embedding) # replace with your own embedding function
        image_tensor = torch.tensor(get_image_features(image_id)) # replace with your own image feature extraction function
        
        # Convert answers to one-hot vectors
        answer_vocab = self.vqa.get_answers_vocab()
        answer_vec = torch.zeros(len(answer_vocab))
        for ans in answers:
            ans_idx = answer_vocab[ans['answer']]
            answer_vec[ans_idx] = 1
        
        return question_tensor, image_tensor, answer_vec

# Instantiate VQA class and VQADataset
vqa = VQA(annotation_file='/path/to/vqa/annotation/file.json')
vqa_dataset = VQADataset(vqa)

# Instantiate DataLoader
batch_size = 32
vqa_dataloader = DataLoader(vqa_dataset, batch_size=batch_size, shuffle=True)