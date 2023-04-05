from cider_scorer import CiderScorer
import json
import sys
sys.path.insert(0, '../dataloader')
from vqa import VQA
import os



class Cider:
    """
    Main Class to compute the CIDEr metric
    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"


if __name__ == "__main__":
    cider = Cider(n=4)

    # config
    ds_path = "../../../../net/projects/ranalab/kh310/vqa"

    for mode in ['val', 'train']:
        print("Mode: ", mode)
        vqa_ds = VQA(annotation_file=os.path.join(ds_path, 'Annotations', f'{mode}.json'))
        gts = {}
        for qa_pair in vqa_ds.dataset:
            ans = [i['answer'] for i in qa_pair['answers']]
            gts[qa_pair['image']] = ans

        FILENAMES = [
            # unimodal baselines
            f'../unimodal_baseline/language/LSTM_outputs_{mode}.json',
            f'../unimodal_baseline/language/T5_outputs_{mode}.json',  # language
            f'../unimodal_baseline/vision/resnet_outputs_{mode}.txt', f'../unimodal_baseline/vision/vit_outputs_{mode}.txt',
            # vision

            # simple multimodal baselines
            f'../multimodal_baseline/clip_outputs_{mode}.txt',
            f'../multimodal_baseline/vit_bert_outputs_{mode}.txt',
            f'../multimodal_baseline/vit_bert_att_HID_DIM_128_outputs_{mode}.txt',

            # competitive multimodal baselines
            f'../competitive_baseline/cross_attention/outputs_{mode}.json',
            f'../competitive_baseline/CLIP/outputs_{mode}.json',
            f'../competitive_baseline/VILT/ViLT_outputs_{mode}.json'
        ]  # competitive multimodal


        for result_fpath in FILENAMES:

            with open(result_fpath, 'r') as f:
                result = json.load(f)

            res = {}

            for group in result['data']:
                res[group['image_id']+'.jpg'] = [group['predicted_answer']]


            score, scores = cider.compute_score(gts, res)

            print(f'{result_fpath}: {score}')
