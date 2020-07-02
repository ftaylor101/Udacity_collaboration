import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(10)  # seed the RNG for all devices


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """
    Actor (Policy) Model.

    Adapted from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py
    """

    def __init__(self, state_size: int = 24, action_size: int = 2, fc1_units: int = 64, fc2_units: int = 128):
        """
        Initialise parameters and build Actor model for DDPG.

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param fc1_units: First fully connected layer size
        :param fc2_units: Second fully connected layer size
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialise the network parameters.
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build an actor (policy) network that maps states -> actions.

        :param x: an input tensor representing state
        :return: the network output representing an action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """
    Critic (Value) Model.

    Adapted from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py
    """

    def __init__(self, state_size: int = 24, action_size: int = 2, fcs1_units: int = 64, fc2_units: int = 128,
                 fc3_units: int = 64):
        """
        Initialize parameters and build model.

        :param state_size: Dimension of each state
        :param action_size: Dimension of each action
        :param fcs1_units: First fully connected layer size prior to state and action concatenation
        :param fc2_units: Second fully connected layer size
        :param fc3_units: Third fully connected layer size
        """
        super().__init__()
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(action_size + fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialise the network parameters.
        """
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Build a critic (value) network that maps (state, action) pairs -> Q-values.
        """
        x = F.relu(self.fcs1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
