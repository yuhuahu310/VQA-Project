from nltk.tokenize import RegexpTokenizer
import re

TOKENIZER = RegexpTokenizer(r'\w+')

def tokenize(s):
    return TOKENIZER.tokenize(s)

def _remove_unused_tokens(s):
    pattern = r'\s*\[unused\d+\]\s*'
    return re.sub(pattern, '', s)

def process_answer(s):
    s = s.replace(' <eos>', '')
    s = _remove_unused_tokens(s)
    return s
