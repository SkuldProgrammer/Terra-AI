import torch
import torch.nn as nn
import torch.nn.functional as F


# FeedForward
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(NeuralNet, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        out = self.lin1(x)
        out = F.relu(out)
        out = self.lin2(out)
        out = F.relu(out)
        out = self.lin3(out)
        return out


class WeatherModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(WeatherModel, self).__init__()
        self.lin1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        y_predicted = self.lin1(x)
        return y_predicted
