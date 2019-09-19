import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from pathlib import Path
import argparse

parser=argparse.ArgumentParser()

parser.add_argument("-epochs", "--epochs", type=int, default=6, help="Enter Number of epochs")
parser.add_argument("-learning_rate", "--learning_rate", type=float, default=0.1, help="Enter Learning Rate")
args=parser.parse_args()

PATH = Path("./model_q1.pth")

learning_rate=args.learning_rate
epochs=args.epochs

transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.5,),(0.5,)),
								])

trainset = datasets.MNIST('~/.pytorch/MNIST_data',download=True,train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

model=nn.Sequential(nn.Linear(784, 128),
					nn.ReLU(),
					nn.Linear(128,64),
					nn.ReLU(),
					nn.Linear(64,10),
					nn.LogSoftmax(dim=1))

losscriterion = nn.NLLLoss()
optimizer=optim.SGD(model.parameters(), lr=learning_rate)

for i in range(epochs):
	running_loss=0
	for images,labels in trainloader:
		images=images.view(images.shape[0],-1)
		optimizer.zero_grad()
		output=model(images)
		loss=losscriterion(output,labels)
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
	print("Training loss:",running_loss/len(trainloader))

torch.save(model, PATH)