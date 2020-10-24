import torch
import json
import random
from model import NeuralNet, WeatherModel
import train
from nltk_utils import bag_of_words, tokenize
import learn_new_things
import speech

device = torch.device("cpu")


def Einlesung():
    global intents, data
    with open("intents.json", "r", encoding="UTF-8") as f:
        intents = json.load(f)
        print("Einlesung erfolgreich!")
        FILE = "Terra-Speak.pth"
        data = torch.load(FILE)


Einlesung()

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
# Defining the model and using it for training
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

while True:
    sentence = input("Du: ")

    if sentence == "schlafen":
        break

    elif sentence == "lernen":
        tag_input = input("Was ist der Oberbegriff? ")
        patterns_input = input("Wie fragt man danach? ")
        responses_input = input("Wie antwortet man darauf, bitte humorvoll, wenn es geht? ")

        learn_new_things.learn(tag_input, patterns_input, responses_input)

    elif sentence == "erweitern":
        tag_input2 = input("Welchen Oberbegriff möchten Sie erweitern? ")
        pattern_input2 = input("Wie fragt man noch danach? ")
        responses_input2 = input("Welche Antwort kann man geben? ")
        learn_new_things.erweitern(tag_input2, pattern_input2, responses_input2)


    elif sentence == "training":
        train.train()
        Einlesung()

    elif sentence == "spezialtraining":
        num_epochs = int(input("Wie viele Epochen soll ich trainieren? "))
        learning_rate = float(input("Welche Lernrate soll ich verwenden? "))
        train.train(num_epochs=num_epochs, learning_rate=learning_rate)

    else:

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        # Wahrscheinlichkeit der Antwort ausrechnen
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        # evaluating tag and response
        if prob.item() > 0.7:
            print(prob.item())
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    text = random.choice(intent['responses'])
                    print(f"Terra: {text}")
                    speech.say(text)

        else:
            print("Terra: Sorry, ich verstehe dich nicht.")
