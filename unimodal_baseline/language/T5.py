import sys
sys.path.insert(0, '../../dataloader')
from dataset import QADataset
from simplet5 import SimpleT5
import pandas as pd
import time
import json
from tqdm import tqdm

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

def train_model(model_type, model_name, train_df, eval_df, source_max_token_len=80, 
                target_max_token_len=25, 
                batch_size=128, max_epochs=5, use_gpu=True,
                load_model_path=''):
    model = SimpleT5()
    if load_model_path:
        model.load_model(model_type, load_model_path, use_gpu=use_gpu)
    else:
        model.from_pretrained(model_type=model_type, model_name=model_name)
    if max_epochs > 0:
        model.train(train_df = train_df, eval_df = eval_df,
                    source_max_token_len=source_max_token_len, 
                    target_max_token_len=target_max_token_len, 
                    batch_size=batch_size, max_epochs=max_epochs, use_gpu=use_gpu)
    return model

def predict(model, questions, answers, print=False):
    targets = []
    predictions = []
    for q, tg in tqdm(list(zip(questions, answers))):
        a = model.predict(q)[0]
        tg = tg.replace(" </s>", "")
        targets.append(tg)
        predictions.append(a)
        if print:
            print('[Question]: {}\n\t[Answer]: {}\n\t[Target]: {}'.format(q, a, tg))
    return questions, targets, predictions


def predict_answers(df, mode, n_samples=30):
    samples = df.sample(n_samples, random_state=int(time.time()))
    print(f"{mode} samples:")
    predict(model, samples['source_text'].tolist(), samples['target_text'].tolist())

def predict_answers_all(model, df, mode, dump=''):
    questions, targets, predictions = predict(
        model,
        df['source_text'].tolist(),
        df['target_text'].tolist(),
        print=False
    )
    imageids = df['imageid'].tolist()
    out = {
        "model_name": "T5",
        "metadata": {
            "mode": mode,
            "modality": ["language"]
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

    if LOAD:
        model = train_model('t5', 't5-small', train, val, max_epochs=0, load_model_path='./outputs/simplet5-epoch-9-train-loss-1.0763-val-loss-1.5209')
    else:
        model = train_model('t5', 't5-small', train, val, max_epochs=30)
    
    # predict_answers(train, "Train")
    # predict_answers(val, "Val")

    predict_answers_all(model, train, 'train', dump='T5_outputs_train.json')
    # predict_answers_all(model, val, 'val', dump='T5_outputs_val.json')
