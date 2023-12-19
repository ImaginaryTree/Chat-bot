import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
from model import CBModel

with open('intents.json', 'r') as f:
    intents = json.load(f)

print(f'\nStage 1 : {intents}\n')

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    print(f'\nStage 2 : {tags}\n')
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
        print(f'\nStage 2.1 : {all_words}')
        print(f'Stage 2.2 : {xy}\n')



ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

print(f'\nStage 3 : {all_words}\n')

all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(f'\nFinal Stage : {all_words}')
print(f'\nFinal Stage : {tags}')

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f'\nTraining set: \nFeature:\n{X_train} \n\nLabel:\n{y_train}')

CBModel(X_train, y_train)

