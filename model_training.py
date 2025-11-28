# CNN Architecture

# importing libraries


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets ,transforms
from torch.utils.data import Dataset, DataLoader
import sys # Added for exit

# tranforming image in tensor and normalizing
# Note: The model is trained *without* normalization.
tranform = transforms.Compose([transforms.Resize((28,28)),
                               transforms.ToTensor()])
# importing data

print("--- MNIST Model Trainer (Custom CNN) ---")
print("Downloading MNIST dataset...")
try:
    train_dataset = datasets.MNIST(root="./data",train=True,transform=tranform,download=True)
    test_dataset = datasets.MNIST(root='./data',train=False,transform=tranform,download=True)
except Exception as e:
    print(f"\nError downloading dataset. Are you connected to the internet?")
    print(f"Details: {e}")
    sys.exit(1)

print("Dataset loaded.")

# converting into dataloader
train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

# CNN Architecture

class CNNModel(nn.Module):
  def __init__(self):
    super(CNNModel,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2) # 28x28 -> 14x14
    self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1 ,padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2) # 14x14 -> 7x7
    self.fc1 = nn.Linear(32*7*7,128) # 32 channels * 7 * 7 image size
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,10)
    self.relu = nn.ReLU()
    # self.sigmoid is defined but not used in the forward pass for 10-class
    # self.sigmoid = nn.Sigmoid() 

  def forward(self ,x):
    x = self.pool1(self.relu(self.conv1(x)))
    x = self.pool2(self.relu(self.conv2(x)))
    x = x.view(x.size(0),-1)  # flatten
    # Note: The original script had no activations here.
    # For a model to learn, it should have them.
    # If you trained with the lines below, keep them.
    # If you trained *without* them, comment them out.
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    # x = self.fc1(x) # Original version (likely a mistake)
    # x = self.fc2(x) # Original version (likely a mistake)
    x = self.fc3(x)
    return x

# Check for CUDA
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

if cuda :
  print("Model will be transferred to CUDA (GPU).")
else:
  print("CUDA is not available. Model will run on CPU.")

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


# train the model
print("Starting training...")
epochs = 50
for epoch in range(epochs):
  model.train()
  running_loss = 0
  for images,labels in train_loader :
    # Move data to the correct device
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    
  print(f"Epoch {epoch+1}/{epochs} , Loss : {running_loss / len(train_loader):.4f}")


# Turn off gradients during evaluation
print("\nStarting evaluation...")
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():  # No gradient calculation
    for images, labels in test_loader:
        # Move data to the correct device
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
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
print("Saving model to 'mnist_cnn_model.pth'")
torch.save(model.state_dict(),'mnist_cnn_model.pth')
print("Done.")
