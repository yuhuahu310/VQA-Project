import numpy as np
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import random
import torch
import clip
from PIL import Image


SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"
def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        # TODO - use torch.nn.Embedding to create the encoding. Initialize dropout layer.
        self.encoding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, x):
        N, S, D = x.shape

        output = x + self.encoding(torch.arange(S, device=x.device, dtype=torch.int)).unsqueeze(0).expand(N, -1, -1)
        output = self.dropout(output)
   
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, word_to_idx, idx_to_word, input_dim, embed_dim, num_heads=4,
                 num_layers=2, max_length=50, device = 'cuda'):
        """
        Construct a new TransformerDecoder instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension of input image feature vectors.
        - embed_dim: Embedding dimension of the transformer.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        print("vocab_size: ", vocab_size)
        self._null = word_to_idx[PAD_TOKEN]
        self._start = word_to_idx.get(SOS_TOKEN, None)
        self.idx_to_word = idx_to_word
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers) 
        
        self.caption_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_length)

        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.image_linear = nn.Linear(512, embed_dim)
        self.text_linear = nn.Linear(512, embed_dim)

        self.score_projection = nn.Linear(embed_dim, vocab_size) 

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def get_data_embeddings(self, features, questions, answers):
        image_features = self.model.float().encode_image(features)
        image_features = self.image_linear(image_features.float())

        # freeze language model
        with torch.no_grad():
            questions_embedding = self.model.float().encode_text(questions)
        
        questions_embedding = self.encoder(questions_embedding)
        questions_embedding = self.text_linear(questions_embedding)

        
        answers_embedding = self.caption_embedding(answers)
        answers_embedding = self.positional_encoding(answers_embedding)

        # Mutimodality Fusion
        multi_feature = image_features * questions_embedding
        multi_feature = torch.unsqueeze(multi_feature, 1)
        return multi_feature, answers_embedding

    def generate_square_subsequent_mask(self, seq_size): 
        mask = (torch.triu(torch.ones(seq_size, seq_size, device=self.device)) == 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, 1).masked_fill(mask == 1, 0)
        return mask
                                      
    def forward(self, features, questions, answers):
        features_embed, captions_embed = self.get_data_embeddings(features, questions, answers)

        mask = self.generate_square_subsequent_mask(captions_embed.shape[1])
        mask.to(captions_embed.dtype)
        
        output = captions_embed
        for layer in self.layers:
            output = layer(output, features_embed, tgt_mask=mask)

        scores = self.score_projection(output)
        return scores

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sample(self, features, questions, max_length=30):
        """
        Given image features and question features, use greedy decoding to predict the answer.
        Inputs:
         - features: image features, of shape (N, D)
         - question: of shape (N, T, D)
         - max_length: maximum possible caption length
        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features).to(self.device)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)
            logits = torch.ones((N, max_length, 16604), device=self.device).float()

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, questions, partial_caption)
                output_logits = output_logits[:, -1, :]
                logits[:, t, :] = output_logits

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions, logits



