import sys
sys.path.insert(0, '../../dataloader')
from dataset import VQADataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, num_layers, dropout):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers)

    def forward(self, sentence):
        embs = self.word_embeddings(sentence)
        _, (hidden, cell) = self.lstm(embs)
        return hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, num_layers, dropout):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embs = self.embedding(input)
        output, (hidden, cell) = self.lstm(embs, (hidden, cell))
        output = self.linear(output.squeeze(0))
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderLSTM, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hidden_size == decoder.hidden_size
        assert encoder.num_layers == decoder.num_layers

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        '''
        :param target of size [target_len, batch_size]
        '''
        batch_size = target.shape[1]
        target_len = target.shape[0]
        outputs = torch.zeros(target_len, batch_size, self.decoder.vocab_size).to(self.device)
        encoder_hidden, encoder_cell = self.encoder(input)

        decoder_input = target[0,:]
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        loss = 0
        for i in range(1, target_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            loss += nn.functional.cross_entropy(decoder_output, target[i])
            outputs[i] = decoder_output
            teacher_forcing = random.random() < teacher_forcing_ratio
            decoder_input = target[i] if teacher_forcing else decoder_output.argmax(dim=1)
            
        return loss

vqa_dataset = VQADataset("../../data", "train")

VOCAB_SIZE = len(vqa_dataset.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = EncoderLSTM(ENC_EMB_DIM, HID_DIM, VOCAB_SIZE, N_LAYERS, ENC_DROPOUT)
dec = DecoderLSTM(DEC_EMB_DIM, HID_DIM, VOCAB_SIZE, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)