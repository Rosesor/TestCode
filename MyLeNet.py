import torch

class LeNet(torch.nn.module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Sequential()


