from vqaTools.vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os
import nltk
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter
from PIL import Image
import seaborn as sns

dataDir='../../'
split = 'val'
annFile='%s/Annotations/%s.json'%(dataDir, split)
imgDir = '%s/Images/%s/' %(dataDir, split)
savefigDir = '%s/Figs/' %(dataDir)	

print(f"Annotations are at {annFile}")
print(f"Images are at {imgDir}")

vqa=VQA(annFile)
# load and display QA annotations for given answer types
"""	
ansTypes can be one of the following
yes/no
number
other
unanswerable
"""
anns = vqa.getAnns()

def answer_length_hist(answers):
    a_lens = [len(tokens) for tokens in answers]
    a_series = pd.Series(a_lens)
    print(f"Here are some statistics about length of answers:\n {a_series.describe()}")
    return a_lens

def lexical_diversity(answers):
    diversity_score = [(len(set(text)) / len(text), len(text)) for text in answers]
    d_df = pd.DataFrame(diversity_score, columns=["lexical_diversity_score", "length_of_answer"])
    s = d_df["lexical_diversity_score"].describe()
    print(f"Here are some statistics about lexical diversity of answers:\n {s}")
    return d_df

def distribution_of_ans():
    ans_types = ["yes/no", "number", "other", "unanswerable"]
    dis = [(tp, len(vqa.getAnns(ansTypes=tp))) for tp in ans_types]
    return dis

def analysis_answers():
    tokenizer = RegexpTokenizer(r'\w+')

    # Distribution of answers based on expected answer type
    temp1 = pd.DataFrame(distribution_of_ans(), columns=["Answer_Type", "Freq"])
    # cur_fig = plt.figure(figsize=(10, 8), dpi=100)
    cur_fig = sns.barplot(temp1,x='Answer_Type',y='Freq',palette='mako')
    # cur_fig.set_yscale("log")
    cur_fig.set_title(f"Answer Type and Its Frequencies")
    fig = cur_fig.get_figure()
    fig.savefig(f"{savefigDir}/answer_type_dist.png")
    plt.close()

    # Retriving all answers
    raw_dist = [j for (i,j) in distribution_of_ans()]
    print(f"Percentage of Unanswerable over all answers: {raw_dist[3]/sum(raw_dist)}")
    answers = [ann["answers"] for ann in anns]
    flat_answers = [item for sublist in answers for item in sublist]
    # print(flat_answers)
    flat_answers = [tokenizer.tokenize(i['answer']) for i in flat_answers]
    # answers = [answer for ann in anns for answer in ann]
    # print(flat_answers)

    # Answer Length
    q_lens = answer_length_hist(flat_answers)
    fig = plt.figure()
    plt.hist(q_lens, bins=50)
    plt.xlabel("answer_length")
    plt.ylabel("freq")
    plt.title("Frequency of Answers Length Histogram")
    # plt.show()
    fig.savefig(f"{savefigDir}/answers_length_hist.png")
    plt.close()

    # lexical diversity kde plot 
    l = lexical_diversity(flat_answers)
    print(l)
    cur_fig = plt.figure(figsize=(10, 8), dpi=100)
    cur_fig = sns.displot(data=l, x="lexical_diversity_score", kde=True)
    cur_fig.set(yscale='symlog')
    fig = cur_fig.fig
    fig.savefig(f"{savefigDir}/answers_lexical_diversity_kde.png")
    plt.close()

    # bivariate kde plot
    cur_fig = plt.figure(figsize=(10, 8), dpi=100)
    cur_fig = sns.displot(data=l, x="length_of_answer", y="lexical_diversity_score",kind="kde", rug=True)
    ax = plt.gca()
    ax.set_xlim([0, 15])
    ax.set_ylim([0.9,1.1])
    fig = cur_fig.fig
    fig.savefig(f"{savefigDir}/answers_lexical_diversity_kde_bivariate.png")
    plt.close()

    return

analysis_answers()