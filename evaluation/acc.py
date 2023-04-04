import json

def get_accuracy(path=''):
    '''
    Takes path to output json file as input
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
    for i in range(n):
        d = data['data'][i]
        ann = annotation[d['image_id'] + '.jpg']
        assert d['image_id'] + '.jpg' == ann['image']
        count, acc = 0, 0
        for j in range(10):
            count += d['predicted_answer'] == ann['answers'][j]['answer']
        for j in range(10):
            acc += min((count - (d['predicted_answer'] == ann['answers'][j]['answer'])) / 3, 1)
        acc_total += acc / 10
    print('Model name: {}\nMode: {}\nAccuracy: {:.4f}'.format(model_name, mode, acc_total / n))
get_accuracy('../competitive_baseline/cross_attention/outputs_val.json')