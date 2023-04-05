from torchtext.data.metrics import bleu_score
from common import tokenize, process_answer, FILENAMES
import json

def build_reference_corpus(mode, unique_only=True):
    with open(f'../data/Annotations/{mode}.json') as f:
        annotations = json.load(f)
    corpus = {}
    for sample in annotations:
        if unique_only:
            ans_set = set(ans['answer'] for ans in sample['answers'])
        else:
            ans_set = [ans['answer'] for ans in sample['answers']]
        ans_tokens_list = [tokenize(process_answer(ans)) for ans in ans_set]
        corpus[sample['image'].replace('.jpg', '')] = ans_tokens_list
    return corpus

def build_candidate_corpus(path):
    with open(path) as f:
        outputs = json.load(f)
    corpus = {sample['image_id']: tokenize(process_answer(sample['predicted_answer'])) for sample in outputs['data']}
    return corpus, outputs['model_name']

def compute_bleu_score(candidate_corpus, reference_corpus):
    cands = []
    refs = []
    for image_id, pred in candidate_corpus.items():
        if image_id in reference_corpus:
            cands.append(pred)
            refs.append(reference_corpus[image_id])
    # breakpoint()
    return bleu_score(cands, refs)

if __name__ == '__main__':
    reference_corpora = {
        'train': build_reference_corpus('train'),
        'val': build_reference_corpus('val')
    }
    # FILENAMES = ['../unimodal_baseline/language/T5_outputs_{mode}.json']
    for path in FILENAMES:
        for mode in ['train', 'val']:
            candidate_corpus, model_name = build_candidate_corpus(path.format(mode=mode))
            score = compute_bleu_score(
                candidate_corpus,
                reference_corpora[mode]
            )
            print(f'Model name: {model_name}, Mode: {mode}, Bleu Score: {score:.4f}')