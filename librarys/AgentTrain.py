import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def relu(x):
  return F.relu(x)

class Agent(nn.Module):
  def __init__(self):
    self.in_dim = 2
    self.hid_dim = 128
    self.value_out_dim = self.hid_dim * 10 * 9
    super(Agent, self).__init__()
    self.cnn1 = nn.Conv2d(self.in_dim, self.hid_dim, kernel_size=3, stride=1, padding=1)
    self.cnn2 = nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, stride=1, padding=1)
    self.cnn3 = nn.Conv2d(self.hid_dim, self.hid_dim, kernel_size=3, stride=1, padding=1)
    self.policy_head = nn.Conv2d(self.hid_dim, 1, kernel_size=1, stride=1, padding=0)
    self.value_head = nn.Linear(self.value_out_dim, 1)

  def forward(self, x):
    x = self.cnn1(x)
    x = relu(x)
    x = self.cnn2(x)
    x = relu(x)
    x = self.cnn3(x)
    x = relu(x)

    x_flat = x.view(x.size(0), -1)

    policy = self.policy_head(x)
    value = self.value_head(x_flat)
    return policy, value




    