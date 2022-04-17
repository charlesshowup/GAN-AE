from torchvision.datasets import FashionMNIST as fm
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize()
])
data_root = '/Users/chenjiayi/Downloads/论文2/data/FasionMNIST'

fm_train = fm(root = data_root, train = True, download = False, transform = img_transform)
train_data = DataLoader(fm_train, batch_size = 100, shuffle = False)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.LeakyReLU(negative_slope=0.01)
        )
    def forward(self,x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder

model = AutoEncoder()
criterion = nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
epoch = 50
losses= []

for e in range(epoch):
    for in_data, _ in train_data:
        in_data = in_data.view(in_data.shape[0], -1)
        in_data = Variable(in_data)
        _, out_data = model(in_data)
        
        loss = criterion(out_data, in_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data)
    print('epoch is: {}, Loss is:{:.4f}'.format(e + 1, loss.data))

torch.save(model, '/Users/chenjiayi/Downloads/论文2/model.pth')
