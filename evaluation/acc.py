import json
from collections import defaultdict
from common import process_answer, FILENAMES

def get_accuracy(path):
    '''
    Input: the output json file of model
    '''
    with open(path) as f:
        data = json.load(f)
    mode = data['metadata']['mode']
    annotation_path = '../data/Annotations/{}.json'.format(mode)
    model_name = data['model_name']
    n = len(data['data'])
    with open(annotation_path) as f:
        annotation = json.load(f)
        annotation = {d['image']:d for d in annotation}
    
    acc_total = 0
    acc_dict = defaultdict(float)
    type_count = defaultdict(int)
    image_ids = set()
    unans_total = 0
    for i in range(n):
        d = data['data'][i]
        if d['image_id'] in image_ids:
            continue
        image_ids.add(d['image_id'])
        ann = annotation[d['image_id'] + '.jpg']
        assert d['image_id'] + '.jpg' == ann['image']
        ans_type = ann['answer_type']
        d['predicted_answer'] = process_answer(d['predicted_answer'])
        count, acc = 0, 0
        for j in range(10):
            count += d['predicted_answer'] == ann['answers'][j]['answer']
        for j in range(10):
            acc += min((count - (d['predicted_answer'] == ann['answers'][j]['answer'])) / 3, 1)
        acc /= 10
        acc_total += acc
        acc_dict[ans_type] += acc
        type_count[ans_type] += 1
        if d['predicted_answer'] == 'unanswerable':
            unans_total += 1
    acc_total /= len(image_ids)
    for ans_type in acc_dict:
        acc_dict[ans_type] /= type_count[ans_type]
    unans_precision = min(type_count['unanswerable'] / unans_total, 1) if unans_total else float('nan')
    print('Model name: {}\n\tMode: {}\n\tOverall accuracy: {:.4f}'.format(model_name, mode, acc_total))
    print('\tAccuracy by answer type: '+', '.join([k + ': ' + f'{v:.4f}' for k, v in acc_dict.items()]))
    print(f'\tPrecision of unanswerable: {unans_precision:.4f}')

if __name__ == '__main__':
    filenames = ['../gated_clip/gated_clip_full_outputs_{mode}.txt'] # FILENAMES can change to selected files
    for path in filenames:
        for mode in ['train', 'val']:
            get_accuracy(path.format(mode=mode))