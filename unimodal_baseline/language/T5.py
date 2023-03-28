import sys
sys.path.insert(0, '../../dataloader')
from dataset import QADataset
from simplet5 import SimpleT5
import pandas as pd
import time

def load_data(path):
    qa_dataset_train = QADataset(path, "train", tokenize=False)
    qa_dataset_val = QADataset(path, "val", tokenize=False)
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

def predict(model, questions, answers):
    for q, tg in zip(questions, answers):
        a = model.predict(q)[0]
        print('[Question]: {}\n\t[Answer]: {}\n\t[Target]: {}'.format(q, a, tg.strip(" </s>")))

def predict_answers(df, mode, n_samples=30):
    samples = df.sample(n_samples, random_state=int(time.time()))
    print(f"{mode} samples:")
    predict(model, samples['source_text'].tolist(), samples['target_text'].tolist())

if __name__ == "__main__":
    train, val = load_data("../../data")
    model = train_model('t5', 't5-small', train, val, max_epochs=0, load_model_path='./outputs/simplet5-epoch-9-train-loss-1.0763-val-loss-1.5209')
    
    predict_answers(train, "Train")
    predict_answers(val, "Val")
