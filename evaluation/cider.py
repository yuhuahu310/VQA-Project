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

        print(len(cider_scorer.crefs))
        print(len(cider_scorer.ctest))

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"


if __name__ == "__main__":
    cider = Cider(n=4)

    # config
    ds_path = "../../../../net/projects/ranalab/kh310/vqa"

    vqa_val = VQA(annotation_file=os.path.join(ds_path, 'Annotations', f'val.json'))
    vqa_train = VQA(annotation_file=os.path.join(ds_path, 'Annotations', f'train.json'))


    result_fpath = '../multimodal_baseline/clip_outputs_val.txt'


    with open(result_fpath, 'r') as f:
        result = json.load(f)

    res = {}
    gts = {}
    for group in result['data']:
        res[group['image_id']+'.jpg'] = [group['predicted_answer']]
    for qa_pair in vqa_ds.dataset:
        ans = [i['answer'] for i in qa_pair['answers']]
        gts[qa_pair['image']] = ans



    score, scores = cider.compute_score(gts, res)

    print(score)
