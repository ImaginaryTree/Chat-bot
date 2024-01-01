import random
import json
import numpy as np
import tensorflow as tf
from model import CBModel
from nltk_utils import bag_of_words, tokenize

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "gfgModel.h5"
data = np.load(FILE, allow_pickle=True)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

device = "cuda"
model = CBModel(input_size, hidden_size, output_size, all_words, tags. model_state).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! type 'quit' to exit.")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = tf.convert_to_tensor(X, dtype=tf.float32)


    output = model(X)
    _, predicted = tf.softmax(output, axis=1)
    tag = tags[predicted.item()]

    probs = tf.softmax(output, dim=1)
    prob = probs(0)[predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intent"]:
                if tag == intent["tag"]:
                    print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
         print(f'{bot_name}: I do not understand...')

    





