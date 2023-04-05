from nltk.tokenize import RegexpTokenizer
import re

TOKENIZER = RegexpTokenizer(r'\w+')
FILENAMES = [
    # unimodal baselines
    '../unimodal_baseline/language/LSTM_outputs_{mode}.json', '../unimodal_baseline/language/T5_outputs_{mode}.json', # language
    '../unimodal_baseline/vision/resnet_outputs_{mode}.txt', '../unimodal_baseline/vision/vit_outputs_{mode}.txt', # vision

    # simple multimodal baselines
    '../multimodal_baseline/clip_outputs_{mode}.txt', '../multimodal_baseline/vit_bert_attn_outputs_{mode}.txt',
    '../multimodal_baseline/vit_bert_outputs_{mode}.txt',
    
    # competitive multimodal baselines
    '../competitive_baseline/cross_attention/outputs_{mode}.json', '../competitive_baseline/CLIP/outputs_{mode}.json',
    '../competitive_baseline/VILT/ViLT_outputs_{mode}.json'
] # competitive multimodal

def tokenize(s):
    return TOKENIZER.tokenize(s)

def _remove_unused_tokens(s):
    pattern = r'\s*\[unused\d+\]\s*'
    return re.sub(pattern, '', s)

def process_answer(s):
    s = s.replace(' <eos>', '')
    s = _remove_unused_tokens(s)
    return s
