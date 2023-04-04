import sys
sys.path.insert(0, '../../dataloader')
from dataset import QADataset
from transformers import ViltProcessor, ViltForQuestionAnswering
import pandas as pd
import time
import json
from tqdm import tqdm
from PIL import Image

def load_data(path, include_imageid=False):
    qa_dataset_train = QADataset(path, "train", tokenize=False, include_imageid=include_imageid)
    qa_dataset_val = QADataset(path, "val", tokenize=False, include_imageid=include_imageid)
    if include_imageid:
        train = pd.DataFrame(list(qa_dataset_train), columns=["source_text", "target_text", "imageid"])
        val = pd.DataFrame(list(qa_dataset_val), columns=["source_text", "target_text", "imageid"])
    else:
        train = pd.DataFrame(list(qa_dataset_train), columns=["source_text", "target_text"])
        val = pd.DataFrame(list(qa_dataset_val), columns=["source_text", "target_text"])
    train['target_text'] = train['target_text'] + " </s>"
    val['target_text'] = val['target_text'] + " </s>"
    return train, val

def predict(mode, model, processor, images, questions, answers, print=False):
    targets = []
    predictions = []
    for i, q, tg in tqdm(list(zip(images, questions, answers))):
        image = Image.open('../../data/'+ mode + "/" + i)
        encoding = processor(image, q[:40], return_tensors="pt")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        a = model.config.id2label[idx]
        tg = tg.replace(" </s>", "")
        targets.append(tg)
        predictions.append(a)
        if print:
            print('[Question]: {}\n\t[Answer]: {}\n\t[Target]: {}'.format(q, a, tg))
    return questions, targets, predictions

def predict_answers_all(model, processor, df, mode, dump=''):
    questions, targets, predictions = predict(
        mode,
        model,
        processor,
        df['imageid'].tolist(), 
        df['source_text'].tolist(),
        df['target_text'].tolist(),
        print=False
    )
    imageids = df['imageid'].tolist()
    out = {
        "model_name": "ViLT",
        "metadata": {
            "mode": mode,
            "modality": ["baseline"]
        },
        "data": []
    }
    assert len(imageids) == len(questions) == len(targets) == len(predictions)
    for img, q, tg, pred in zip(imageids, questions, targets, predictions):
        out["data"].append({
            "image_id": img.replace(".jpg", ""),
            "question": q,
            "predicted_answer": pred,
            "target_answer": tg
        })
    if dump:
        with open(dump, 'w') as f:
            json.dump(out, f, indent=4)
    return imageids, questions, targets, predictions


if __name__ == "__main__":
    EVAL = True

    train, val = load_data("../../data", include_imageid=EVAL)

    LOAD = True
    if EVAL: LOAD = EVAL
    # breakpoint()
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    
    # predict_answers(train, "Train")
    # predict_answers(val, "Val")

    predict_answers_all(model, processor, train, 'train', dump='ViLT_outputs_train.json')
    # predict_answers_all(model, processor, val, 'val', dump='ViLT_outputs_val.json')
