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

def question_length_hist(questions):
    q_lens = [len(tokens) for tokens in questions]
    q_series = pd.Series(q_lens)
    print(f"Here are some statistics about length of questions:\n {q_series.describe()}")
    return q_lens

def top_n_popular_first_words(questions, n):
    first_words = [tokens[0].lower() for tokens in questions]
    words_counts = Counter(first_words).most_common(n)
    df = pd.DataFrame(words_counts, columns =['word', 'freq'])
    return df

def lexical_diversity(questions):
    diversity_score = [(len(set(text)) / len(text), len(text)) for text in questions]
    d_df = pd.DataFrame(diversity_score, columns=["lexical_diversity_score", "length_of_question"])
    s = d_df["lexical_diversity_score"].describe()
    print(f"Here are some statistics about lexical diversity of questios:\n {s}")
    return d_df
    
def analysis_questions():
    tokenizer = RegexpTokenizer(r'\w+')
    questions = [tokenizer.tokenize(ann['question']) for ann in anns]
    # print(anns)

    # Question Length
    q_lens = question_length_hist(questions)
    fig = plt.figure()
    plt.hist(q_lens, bins=50)
    plt.xlabel("question_length")
    plt.ylabel("freq")
    plt.title("Frequency of Question Length Histogram")
    # plt.show()
    fig.savefig(f"{savefigDir}/questions_length_hist.png")
    plt.close()

    # N Most popular first words
    n = 20
    words_counts_df = top_n_popular_first_words(questions, n)
    print(words_counts_df)
    cur_fig = plt.figure(figsize=(10, 8), dpi=100)
    cur_fig = sns.barplot(words_counts_df,x='word',y='freq',palette='mako')
    cur_fig.set_yscale("log")
    cur_fig.set_title(f"Top {n} most popular first words with frequencies")
    fig = cur_fig.get_figure()
    fig.savefig(f"{savefigDir}/most_popular_{n}_first_words_bar.png")
    plt.close()
    
    # lexical diversity kde plot 
    l = lexical_diversity(questions)
    print(l)
    cur_fig = plt.figure(figsize=(10, 8), dpi=100)
    cur_fig = sns.displot(data=l, x="lexical_diversity_score", kde=True)
    cur_fig.set(yscale='log')
    fig = cur_fig.fig
    fig.savefig(f"{savefigDir}/questions_lexical_diversity_kde.png")
    plt.close()

    # bivariate kde plot
    cur_fig = plt.figure(figsize=(10, 8), dpi=100)
    cur_fig = sns.displot(data=l, x="length_of_question", y="lexical_diversity_score",kind="kde", rug=True)
    fig = cur_fig.fig
    fig.savefig(f"{savefigDir}/questions_lexical_diversity_kde_bivariate.png")
    plt.close()

analysis_questions()