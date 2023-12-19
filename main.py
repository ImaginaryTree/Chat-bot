import random
import json
from nltk_utils import bag_of_words, tokenize


with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

