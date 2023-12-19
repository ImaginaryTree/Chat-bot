import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentences):
    return nltk.word_tokenize(sentences)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentences, all_words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentences]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag





#! Testing

# text = "My name is Fadhlan, How about you?"
# print(text)
# text = tokenize(text)
# print(text)

# text1 = ["organize", "organizes", "organizing"]
# print(text1)
# text1_stem = [stem(w) for w in text1]
# print(text1_stem)