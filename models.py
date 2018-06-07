import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 7)
        self.conv2 = nn.Conv2d(8, 10, 5)
        self.fc1 = nn.Linear(10 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 3 * 2)

        self.to_identity_transformation(self.fc2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = F.relu(self.fc1(x.view(-1, 10 * 3 * 3)))
        theta = self.fc2(x).view(-1, 2, 3)

        return theta

    @staticmethod
    def to_identity_transformation(fc):
        fc.weight.data.zero_()
        fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization_net = LocalizationNet()

    def forward(self, u):
        theta = self.localization_net(u)
        grid = F.affine_grid(theta, u.shape)
        v = F.grid_sample(u, grid)

        return v


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 10)
        self.stn = SpatialTransformerNetwork()

    def forward(self, x):
        x = self.stn(x)

        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), kernel_size=2, stride=2)
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.log_softmax(x, dim=1)
