import nltk
from nltk.stem.porter import PorterStemmer

# nltk.download("punkt")
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag(tokenized_sentence, all_words):
    pass

