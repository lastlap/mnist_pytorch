import torch
from torch import nn
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt

PATH = Path("./model_q1.pth")

# model=nn.Sequential(nn.Linear(784, 128),
#                     nn.ReLU(),
#                     nn.Linear(128,64),
#                     nn.ReLU(),
#                     nn.Linear(64,10),
#                     nn.LogSoftmax(dim=1))

model = torch.load(PATH)
model.eval()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
                                ])

trainset = datasets.MNIST('~/.pytorch/MNIST_data',download=True,train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)

images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
val=(ps==(torch.max(ps))).nonzero()[0,1]
print('Number predicted:',int(val))
print(plt.imshow(img.resize_(1, 28, 28).numpy().squeeze()))
plt.show()