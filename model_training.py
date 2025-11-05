# CNN Architecture

# importing libraries

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets ,transforms
from torch.utils.data import Dataset, DataLoader

# tranforming image in tensor and normalizing
tranform = transforms.Compose([transforms.Resize((28,28)),
                               transforms.ToTensor()])
# importing data

train_dataset = datasets.MNIST(root="./data",train=True,transform=tranform,download=True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=tranform,download=True)

# converting into dataloader

train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

# CNN Architecture

class CNNModel(nn.Module):
  def __init__(self):
    super(CNNModel,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
    self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1 ,padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
    self.fc1 = nn.Linear(32*7*7,128)
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,10)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self ,x):
    x = self.pool1(self.relu(self.conv1(x)))
    x = self.pool2(self.relu(self.conv2(x)))
    x = x.view(x.size(0),-1)  # flatten
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x

cuda = torch.cuda.is_available()
if cuda :
  model = model.cuda()
  print("Model transferred to CUDA")
else:
  print("CUDA is not available. Model will run on CPU.")

model = CNNModel()
if cuda:
    model = model.to('cuda')
else:
    model = model.to('cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


# train the model

epochs = 10
for epoch in range(epochs):
  model.train()
  running_loss = 0
  for images,labels in train_loader :
    optimizer.zero_grad()
    outputs = model(images.to('cuda'))
    loss = criterion(outputs,labels.to('cuda'))
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  print(f"Epoch {epoch+1}/{epochs} , Loss : {loss.item()}")


  # Turn off gradients during evaluation
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():  # No gradient calculation
    for images, labels in test_loader:
        if cuda:
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)  # optional, to calculate test loss
        test_loss += loss.item()

        # Get predicted class with highest score
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Compute accuracy and average loss
accuracy = 100 * correct / total
avg_test_loss = test_loss / len(test_loader)

print(f"\nTest Accuracy: {accuracy:.2f}%")
print(f"Average Test Loss: {avg_test_loss:.4f}")


# save train model
torch.save(model.state_dict(),'mnist_cnn_model.pth')

