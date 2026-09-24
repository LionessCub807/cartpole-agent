import torch
from torch import nn
import torch.nn.functional as F

# Deep Q-Learning Model
class DQN(nn.Module):

    # Initialization function for number of layers and neurons
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        # define the hidden layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    # Implement the activation functions and forward
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == '__main__':
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(1, state_dim)
    output = net(state)
    print(output)
                