import os
import torch
import imageio.v3 as imageio
import torchvision

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet34
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, classification_report 
import seaborn as sns
import onnx
import onnxruntime




# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Data
raw_data_train = r'E:\ITS 365 Machine Learning\ITS_365_Project\Car Body Type Data\train'  

raw_data_test  = r'E:\ITS 365 Machine Learning\ITS_365_Project\Car Body Type Data\test' 

# Train data
dataset_train = []
labels_train  = []
targets_train = []

# Transform train data into tensor
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # Resize Images to the same dimensions 
    transforms.ToTensor()
])
# Load from folders
for folder in os.listdir( raw_data_train ):
    ## print(folder)
    for image in os.listdir( os.path.join(raw_data_train, folder) ):
        if folder not in labels_train:
            labels_train.append( folder )
        targets_train.append(  labels_train.index(folder)  )
        img_arr = imageio.imread(  os.path.join(raw_data_train, folder, image), pilmode="RGB"  )
        
        img_tensor = train_transform(img_arr) 
        dataset_train.append(img_tensor)


data_train_tensor = torch.stack( dataset_train )
targets_train = torch.Tensor(  targets_train  ).type(   torch.LongTensor   )

# Calculate mean and standard deviation per channel




# Redefine transform with normalization
normalized_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Reprocess the training images using normalization
dataset_train = []
for folder in os.listdir(raw_data_train):
    for image in os.listdir(os.path.join(raw_data_train, folder)):
        img_arr = imageio.imread(os.path.join(raw_data_train, folder, image), pilmode="RGB")
        img_tensor = normalized_train_transform(img_arr)
        dataset_train.append(img_tensor)

data_train_tensor = torch.stack(dataset_train)

len(labels_train)
        
len( targets_train )

print(dataset_train[3].shape)

print(targets_train.shape)

# Display Distribution of Training Data
y_train_np = targets_train.numpy() 
y_train_np.shape

the_set = np.unique(  y_train_np  )
the_set

_ = plt.hist( targets_train.numpy() , bins="auto" )
plt.show()

# Test Data
dataset_test = []
labels_test = []
targets_test = []

# Transforms Test data into tensor
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Load from folders
for folder in os.listdir(raw_data_test):
    for image in os.listdir(os.path.join(raw_data_test, folder)):
        if folder not in labels_test:
            labels_test.append(folder)
        targets_test.append(labels_test.index(folder))
        img_arr = imageio.imread(os.path.join(raw_data_test, folder, image), pilmode="RGB")
        img_tensor = test_transform(img_arr)
        dataset_test.append(img_tensor)

data_test_tensor = torch.stack(dataset_test)
targets_test = torch.tensor(targets_test, dtype=torch.long)

print("Test sample shape:", dataset_test[0].shape)  # Display Test Data 
print("Test targets size:", targets_test.shape[0])

X_train = data_train_tensor  # Already float32
y_train = targets_train      # Already long
X_test = data_test_tensor
y_test = targets_test


print(X_train.dtype)  # Should be torch.float32
print(y_train.dtype)  # Should be torch.int64 (Long)


train_dataset = TensorDataset(data_train_tensor, targets_train)
test_dataset  = TensorDataset(data_test_tensor, targets_test)

# DataLoader 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)



# Helper Layer: Flatten input tensor
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(self.shape)
    
# CNN Model
class CNN_Model_Cars(nn.Module):
    def __init__(self, input_channels=3, num_classes=3, lr=0.001, weight_decay=0.001, class_weights=None):
        super(CNN_Model_Cars, self).__init__()

        self.learning_rate = lr
        self.weight_decay = weight_decay

        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.leaky_relu = nn.LeakyReLU()

        # Fully connected layers with an additional hidden layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc_hidden = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Loss (supports optional class weights)
        if class_weights is not None:
            self.loss_function = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_function = nn.CrossEntropyLoss()

        # Optimizer: Adam
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc_hidden(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x




# Train & Evaluate Model
model = CNN_Model_Cars().to(device)
N_Epochs = 200
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

all_preds = []
all_targets = []

for epoch in range(N_Epochs):  
    train_loss = 0
    train_acc = 0
    model.train()  # Set model to training mode
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)  # Get model outputs
        loss = model.loss_function(outputs, targets)  # Calculate loss
        acc = (outputs.argmax(dim=1) == targets).float().mean()  # Calculate accuracy
        
        model.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagate
        model.optimizer.step()  # Update weights

        train_loss += loss.item()
        train_acc += acc.item()

    val_loss = 0
    val_acc = 0
    model.eval()  # Set the model to evaluation mode
    all_preds = []  # Reset predictions
    all_targets = []  # Reset targets
    with torch.no_grad():  # Disable gradient calculation during evaluation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = model.loss_function(outputs, targets)
            acc = (outputs.argmax(dim=1) == targets).float().mean()
            
            val_loss += loss.item()
            val_acc += acc.item()

            # Collect predictions and targets for metrics
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    val_loss /= len(test_loader)
    val_acc /= len(test_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Print only every 10 epochs or on the last epoch
    if (epoch + 1) % 10 == 0 or epoch + 1 == N_Epochs:
        print(f"Epoch {epoch+1}/{N_Epochs} => Train Loss: {train_loss:.7f}, Train Acc: {train_acc:.7f} | Val Loss: {val_loss:.7f}, Val Acc: {val_acc:.7f}")

# Plotting Loss and Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Validation Loss", marker="o")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy", marker="o")
plt.plot(val_accuracies, label="Validation Accuracy", marker="o")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# Print Classification Report
print("Classification Report:")
print(classification_report(all_targets, all_preds, target_names=labels_train))

# Confusion Matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_train, yticklabels=labels_train)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# Export to ONNX
onnx_file_path = "car_body_model.onnx"

# Dummy input to trace the model (batch size of 1, 3 channels, 64x64 image)
dummy_input = torch.randn(1, 3, 64, 64).to(device)

# Export the model
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_file_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Model has been exported to ONNX format at: {onnx_file_path}")


# Load the model using ONNX
onnx_model = onnx.load(onnx_file_path)

# Check if the model is valid (optional, but good practice)
onnx.checker.check_model(onnx_model)
print("ONNX model loaded and verified!")

# Create a runtime session for ONNX inference
ort_session = onnxruntime.InferenceSession(onnx_file_path)

# Prepare dummy input data (same shape as the model expects)
dummy_input = torch.randn(1, 3, 64, 64).to(device)  # Example dummy input (batch size = 1)

# Convert dummy input to numpy
input_data = dummy_input.cpu().numpy()

# Run inference with ONNX Runtime
ort_inputs = {ort_session.get_inputs()[0].name: input_data}
ort_outputs = ort_session.run(None, ort_inputs)

# Display the results
print("Labels: ", labels_train)  # Assuming 'labels_train' contains the labels in your training dataset
print("ONNX Runtime outputs: ", ort_outputs)








