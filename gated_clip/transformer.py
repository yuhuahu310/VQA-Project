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
import sys
sys.path.insert(0, '../dataloader')
from dataset import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, OOV_TOKEN, OCR_TOKEN
from contextlib import nullcontext


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
                 num_layers=2, max_length=50, device = 'cuda', fusion='mult', quality_detector=None):
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
        self.vocab_size = vocab_size
        print("vocab_size: ", vocab_size)
        self._null = word_to_idx[PAD_TOKEN]
        self._start = word_to_idx.get(SOS_TOKEN, None)
        self._unanswerable = word_to_idx.get("unanswerable", None)
        self._end = word_to_idx[EOS_TOKEN]
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

        self.answer_type_head = nn.Sequential(
            nn.Linear(embed_dim + 8 if quality_detector else embed_dim, 128),
            nn.Linear(128, 4)
        )

        self.score_projection = nn.Linear(embed_dim, vocab_size)

        # multimodal fusion
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

        assert fusion == 'mult' or fusion == 'cross_attn'
        self.fusion = fusion

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)
        self.quality_detector = quality_detector

    def get_data_embeddings(self, features, questions, answers, freeze_encoders=True):
        with torch.no_grad() if freeze_encoders else nullcontext():
            image_features = self.model.float().encode_image(features)
            # freeze language model
            questions_embedding = self.model.float().encode_text(questions)
        image_features = self.image_linear(image_features.float())

        
        questions_embedding = self.encoder(questions_embedding)
        questions_embedding = self.text_linear(questions_embedding)

        
        answers_embedding = self.caption_embedding(answers)
        answers_embedding = self.positional_encoding(answers_embedding)

        # Multimodality Fusion
        if self.fusion == 'mult':
            multi_feature = image_features * questions_embedding
        else: # attention
            multi_feature, attn_output_weights = self.multihead_attn(query=questions_embedding, key=image_features, value=image_features)
        multi_feature = torch.unsqueeze(multi_feature, 1)
        return multi_feature, answers_embedding

    def generate_square_subsequent_mask(self, seq_size): 
        mask = (torch.triu(torch.ones(seq_size, seq_size, device=self.device)) == 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, 1).masked_fill(mask == 1, 0)
        return mask
                                      
    def forward(self, features, questions, answers, freeze_encoders=True):
        features_embed, captions_embed = self.get_data_embeddings(features, questions, answers, freeze_encoders=freeze_encoders)

        # Only compute answer type while training
        if self.quality_detector:
            quality_vector = self.quality_detector(self.quality_detector.transform(features))
            answer_type = self.answer_type_head(torch.cat((quality_vector, features_embed.squeeze(1)), dim=1))
        else:
            answer_type = self.answer_type_head(features_embed.squeeze(1))

        mask = self.generate_square_subsequent_mask(captions_embed.shape[1])
        mask.to(captions_embed.dtype)
        
        output = captions_embed
        for layer in self.layers:
            output = layer(output, features_embed, tgt_mask=mask)

        scores = self.score_projection(output)
        return scores, answer_type
    
    def forward_answer_type(self, features, questions, answers):
        features_embed, _ = self.get_data_embeddings(features, questions, answers)
        if self.quality_detector:
            quality_vector = self.quality_detector(self.quality_detector.transform(features))
            answer_type = self.answer_type_head(torch.cat((quality_vector, features_embed.squeeze(1)), dim=1))
        else:
            answer_type = self.answer_type_head(features_embed.squeeze(1))
        return answer_type

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

            
            predicted_answer_type_logits = self.forward_answer_type(features, questions, torch.ones(N, device=self.device).to(torch.long).unsqueeze(1))
            predicted_answer_type = torch.argmax(predicted_answer_type_logits, dim=-1)
            answer_type_mask = predicted_answer_type != 0
            n = answer_type_mask.sum()
            

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((n, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(n, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [n] -> [n, 1]
            partial_caption = partial_caption.unsqueeze(1)
            logits = torch.ones((n, max_length, self.vocab_size), device=self.device).float()

            features = features[answer_type_mask, :]
            questions = questions[answer_type_mask, :]

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits, _ = self.forward(features, questions, partial_caption)
                output_logits = output_logits[:, -1, :]
                logits[:, t, :] = output_logits

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)
            
            final_captions = self._null * np.ones((N, max_length), dtype=np.int32)
            final_captions[answer_type_mask.cpu().numpy(), :] = captions
            unanswerable_answers = np.array([self._start, self._unanswerable, self._end] + [self._null] * (max_length - 3), 
                                                dtype=np.int32)
            final_captions[~answer_type_mask.cpu().numpy(), :] = unanswerable_answers
            final_logits = torch.ones((N, max_length, self.vocab_size), device=self.device).float()
            final_logits[answer_type_mask, :] = logits
            unanswerable_logits = torch.zeros((1, max_length, self.vocab_size), device=self.device)
            tmp = F.one_hot(torch.tensor([self._unanswerable, self._null, self._end], device=self.device), self.vocab_size)
            unanswerable_logits[:, 0, :] = tmp[0]
            unanswerable_logits[:, 1, :] = tmp[2]
            unanswerable_logits[:, 2:, :] = tmp[1]
            final_logits[~answer_type_mask, :] = unanswerable_logits

            # caption (N, 8), logits (N, vocab_size), answer_type_logits (N, 4)
            return final_captions, final_logits, predicted_answer_type_logits



