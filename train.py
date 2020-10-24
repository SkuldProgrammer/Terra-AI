import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open("intents.json", encoding="UTF-8") as json_file:
    intents = json.load(json_file)


def train(num_epochs=500, learning_rate=0.001):
    global intents
    # Use of a JSON-File to read trough the training data
    with open("intents.json", "r", encoding="UTF-8") as f:
        intents = json.load(f)

    # Will hold every word to tokenize and stem them
    all_words = []

    # Will hold every tag to classify the words
    tags = []

    # Will hold patterns and tags
    xy = []

    # the JSON-file is treated like a dictionary, therefore we have to use a key for the loop
    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            # We don´t want to have lists in the all_words list, therefore we extend instead of appending them
            all_words.extend(w)
            # to be able to link the words to the different tags
            xy.append((w, tag))

    # setting up the excluded characters
    ignore_words = ["?", "!", ".", ","]
    all_words = [stem(w) for w in all_words if w not in ignore_words]

    # getting a alphabetically sorted list without duplicate words (function of set)
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    Y_train = []

    for pattern_sentence, tag in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)

        # Get the index of the tag of the tags-list
        label = tags.index(tag)
        Y_train.append(label)  # CrossEntropyLoss

    # Create np.arrays, arrays with only zeros with the length of corresponding data
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Dataset-Class to train it easily
    class ChatDataSet(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = Y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    # Hyperparameters
    batch_size = 8
    hidden_size = 80
    output_size = len(tags)
    input_size = len(all_words)

    # Creating a custom data-set to feed into the neural network
    dataset = ChatDataSet()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Checking if working with gpu is available
    device = torch.device("cpu")

    # Defining the model and using it for training
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(device, torch.int64)

            # forward
            outputs = model(words)
            loss = criterion(outputs, labels)

            # backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print("Epoch " + str(epoch) + " finished! " + f"loss={loss.item():.4}" + "\n " + str(num_epochs - epoch)
                  + " remaining!")

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "Terra-Speak.pth"
    torch.save(data, FILE)

    print(f"Training complete! Model named {FILE} saved.")


if __name__ == "__main__":
    train()
