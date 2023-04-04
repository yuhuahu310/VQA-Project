import json
from collections import defaultdict

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
    for i in range(n):
        d = data['data'][i]
        ann = annotation[d['image_id'] + '.jpg']
        assert d['image_id'] + '.jpg' == ann['image']
        ans_type = ann['answer_type']
        d['predicted_answer'] = d['predicted_answer'].replace(' <eos>', '') # if eos token exists
        count, acc = 0, 0
        for j in range(10):
            count += d['predicted_answer'] == ann['answers'][j]['answer']
        for j in range(10):
            acc += min((count - (d['predicted_answer'] == ann['answers'][j]['answer'])) / 3, 1)
        acc /= 10
        acc_total += acc
        acc_dict[ans_type] += acc
        type_count[ans_type] += 1
    acc_total /= n
    for ans_type in acc_dict:
        acc_dict[ans_type] /= type_count[ans_type]
    print('Model name: {}\n\tMode: {}\n\tOverall accuracy: {:.4f}'.format(model_name, mode, acc_total))
    print('\tAccuracy by answer type: '+', '.join([k + ': ' + f'{v:.4f}' for k, v in acc_dict.items()]))

if __name__ == '__main__':
    filenames = ['../unimodal_baseline/language/LSTM_outputs_{mode}.json', '../unimodal_baseline/language/T5_outputs_{mode}.json', # language baselines
                '../unimodal_baseline/vision/resnet_outputs_{mode}.txt', '../unimodal_baseline/vision/vit_outputs_{mode}.txt', # vision baselines
                '../multimodal_baseline/clip_outputs_{mode}.txt', # simple multimodal
                '../competitive_baseline/cross_attention/outputs_{mode}.json', '../competitive_baseline/CLIP/outputs_{mode}.json', # competitive multimodal
                '../competitive_baseline/VILT/ViLT_outputs_{mode}.json'] # competitive multimodal
    for path in filenames:
        for mode in ['train', 'val']:
            get_accuracy(path.format(mode=mode))