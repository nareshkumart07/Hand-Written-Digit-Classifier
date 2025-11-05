import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset , DataLoader
from torchvision import Datasets ,transforms

X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
Y_train_tensor = torch.tensor(y_train,dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test,dtype=torch.float32)
Y_test_tensor = torch.tenosr(y_test,dtype=torch.float32).unsueeze(1)

train_set = Dataset(X_train_tensor,Y_train_tensor)
test_set = Dataset(X_test_tensor,Y_test_tensor)

train_loader = DataLoader(train_set,batch_size=128,shuffle=True)
test_loader = DataLoader(test_set,batch_size=64,shuffle=False)


class NerualNetwork(nn.Module):
  def __init__(self,input_size):
    super(NeuralNetwork,self).__init__()
    self.layer1 = nn.Linear(input_size,64)
    self.layer2 = nn.Linear(64,32)
    self.layer3 = nn.Linear(32,16)
    self.layer4 = nn.Linear(16,1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.sigmoid()

  def forward(self,x):
    x = self.relu(self.layer1)
    x = self.relu(self.layer2)
    x = self.relu(self.layer3)
    x = self.sigmoid(self.layer4)
    return x


# defining model

model = NerualNetwork(X_train.shape[1])
criterion = nn.BCELoss
optimizer = optim.Adam(model.parameters(),lr=0.001)

# train
epochs =100

for epoch in epochs :
  model.train()
  for inputs ,labels in train_loader :
    optimizer.zero_grad()
    ouptus = model(inputs)
    loss = criterion(outputs ,labels)
    loss.backward()
    optimizer.step()

# test

with torch.no_grad(): # use for no gradient calculation
  y_pred = []
  y_true = []
  model.eval()
  for inputs ,labels in test_loader:
    predic = model(inputs)
    y_pred.extend(predic.round().squeeze().tolist())
    y_true.extend(labels.squeeze().tolist())


print(accuracy_score(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(recall_score(y_true,y_pred))


from google.colab import files
import os

# Prompt the user to upload a file
uploaded = files.upload()

# Get the filename of the uploaded file
if uploaded:
    image_filename = list(uploaded.keys())[0]
    print(f"Uploaded file: {image_filename}")

    # Assuming the uploaded file is in the current directory
    image_path = os.path.join(os.getcwd(), image_filename)

    # Use the predict_image_class function from the previous cell
    # Make sure 'model', 'classes', and 'cuda' are defined in your environment
    if 'model' in globals() and 'classes' in globals() and 'cuda' in globals():
        try:
            predicted_class = predict_image_class(image_path, model, classes, cuda)
            print(f"Predicted Class for {image_filename}: {predicted_class}")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
    else:
        print("Required variables (model, classes, cuda) are not defined. Please run the previous cells.")

else:
    print("No file was uploaded.")
