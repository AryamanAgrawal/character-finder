from nltk import pos_tag, word_tokenize
from collections import Counter
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def read_text():
    with open('./moby.txt', mode='r') as f:
        book = f.read()
    f.close()
    return book


def text_tokenize(book):
    tokenize = word_tokenize(book)
    return tokenize


def tagging(tokenize):
    tagged_text = pos_tag(tokenize)
    return tagged_text


def find_proper_nouns(tagged_text):
    proper_nouns = []
    i = 0
    while i < len(tagged_text):
        if tagged_text[i][1] == 'NNP':
            if tagged_text[i+1][1] == 'NNP':
                proper_nouns.append(
                    tagged_text[i][0].lower() + " " + tagged_text[i+1][0].lower())
                i += 1
            else:
                proper_nouns.append(tagged_text[i][0].lower())
        i += 1
    return proper_nouns


def summarize_text(proper_nouns, top_num):
    counts = dict(Counter(proper_nouns).most_common(top_num))
    return counts


book = read_text()
tokenize = text_tokenize(book)
tagged_text = tagging(tokenize)
proper_nouns = find_proper_nouns(tagged_text)
sum = summarize_text(proper_nouns, 20)
print(sum)
freq = nltk.FreqDist(sum)
freq.plot(20)
