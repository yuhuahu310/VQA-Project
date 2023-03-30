
import torch
# from utils import * 
from torch.utils.data import DataLoader
from trainer import Trainer
from transformer import TransformerDecoder
from matplotlib import pyplot as plt
import random
import numpy as np
import sys
sys.path.insert(0, '../../dataloader')
from dataset import VQADataset, collate_fn_pad_image


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

set_all_seeds(42)
exp_name = 'case1'

train_dataset = VQADataset("../../data", "train")
qa_dataset_val = VQADataset("../../data", "val")

VOCAB_SIZE = len(train_dataset.vocab)
ENC_EMB_DIM = 1000
DEC_EMB_DIM = 256
HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 512
NUM_EPOCH = 100


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad_image, num_workers=64)
val_dataloader = DataLoader(qa_dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad_image, num_workers=64)

device = 'cuda'
transformer = TransformerDecoder(
          word_to_idx=train_dataset.vocab,
          idx_to_word = train_dataset.reverse_vocab,
          input_dim=ENC_EMB_DIM,
          embed_dim=HID_DIM,
          num_heads=4,
          num_layers=6,
          max_length=30,
          device = device
        )

trainer = Trainer(transformer, train_dataloader, val_dataloader,
          num_epochs=100,
          learning_rate=1e-4,
          device = device
        )

trainer.train()

# torch.save(transformer.state_dict(), "/home/yizhi/VQA-Project/unimodal_baseline/vision/model.pth")

# transformer.load_state_dict(torch.load("/home/yizhi/VQA-Project/unimodal_baseline/vision/model_99.pth"))
# transformer.eval()


# Plot the training losses.
plt.plot(trainer.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
os.makedirs('plots', exist_ok=True)
plt.title('Training loss history')
plt.savefig('plots/' + exp_name + '_loss_out.png')

def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != PAD_TOKEN:
                words.append(word)
            if word == EOS_TOKEN:
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded

def vis_imgs(split):
    # data = {'train': train_dataset.data, 'val': val_dataset.data}[split]
    loader = {'train': train_dataloader, 'val': val_dataloader}[split]
    num_imgs = 0 
    for batch in loader:
      features, questions, answers = batch
      
      gt_captions = decode_captions(answers.numpy(), train_dataset.reverse_vocab)
      sample_captions = transformer.sample(features.float(), max_length=30)
      sample_captions = decode_captions(sample_captions, train_dataset.reverse_vocab)
      
      for gt_caption, sample_caption, img in zip(gt_captions, sample_captions, features):
          # Skip missing URLs.
          if img is not None: 
            img = img.swapaxes(0, 1)
            img = img.swapaxes(1, 2)
            plt.imshow(img)            
            plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.savefig('plots/' + exp_name + '_%s_%d.png' % (split, num_imgs))
            num_imgs += 1
            if num_imgs >= 5: break
      return 

vis_imgs('train')
vis_imgs('val')