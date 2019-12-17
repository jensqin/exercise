import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import altair as alt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# linear regression
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

traindf = pd.DataFrame({'x':x_train.T.squeeze(), 'y':y_train.T.squeeze()})
traindata = torch.utils.data.DataLoader(
    dataset=traindf, batch_size=5, shuffle=True
)

model = nn.Linear(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

prediction = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(y_train, prediction - y_train, 'ro')
plt.xlabel('y')
plt.ylabel(r"$\hat{y}$", rotation=0)
plt.title('residual plot')

# logistic regression
batch_size = 64
learning_rate = 1e-3
num_epochs = 100

train_dataset = datasets.FashionMNIST(
    root='../data', train=True, transform=transforms.ToTensor(), 
    download=True
)
test_dataset = datasets.FashionMNIST(
    root='../datas', train=False, transform=transforms.ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class logitReg(nn.Module):
    def __init__(self, indim, nclass):
        super(logitReg, self).__init__()
        self.logistic = nn.Linear(indim, nclass)

    def forward(self, x):
        out = self.logistic(x)
        return out

model = logitReg(28 ** 2, 10)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print('*' * 10)
    print(f'Epoch {epoch + 1}')
    since = time.time()
    run_loss = 0.0
    run_acc = 0.0
    model.train()
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)
        out = model(img)
        loss = criterion(out, label)
