import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentences):
    return nltk.word_tokenize(sentences)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentences, all_words):
    pass

# text = "My name is Fadhlan, How about you?"
# print(text)
# text = tokenize(text)
# print(text)

# text1 = ["organize", "organizes", "organizing"]
# print(text1)
# text1_stem = [stem(w) for w in text1]
# print(text1_stem)