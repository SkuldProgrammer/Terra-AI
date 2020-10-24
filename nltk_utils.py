import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
stemmer = LancasterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # A "list" with the length of all the words is created in order to change the 0 at the given index to 1 if the word
    # is in all_words
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0

    return bag


