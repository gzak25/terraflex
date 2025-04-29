import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from cnn import CNN


# Dataset class
class SensorDataset(Dataset):
    def __init__(self, folder_path, label_mapping):
        self.data = []
        self.labels = []
        self.label_mapping = label_mapping

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                label_str = filename.split("_")[0]
                label = self.label_mapping[label_str]

                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)

                sensor_data = df[
                    [
                        "emg",
                        "front_left_sensor_1",
                        "front_left_sensor_2",
                        "front_right_sensor_1",
                        "front_right_sensor_2",
                        "back_left_sensor",
                        "back_right_sensor",
                    ]
                ].values

                # Transpose to (channels, sequence_length)
                sensor_data = torch.tensor(sensor_data, dtype=torch.float).T

                self.data.append(sensor_data)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Setup

# Path to CSV folder
FOLDER_PATH = "EMG_and_pressure_data"

# Label mapping
LABEL_MAPPING = {"stable": 0, "forward": 1, "right": 2, "left": 3, "back": 4}

# Hyperparameters
NUM_CLASSES = 5
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_PATH = "model.pth"

# Load dataset
dataset = SensorDataset(FOLDER_PATH, LABEL_MAPPING)

# Split into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        inputs = inputs.float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(
        f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%"
    )

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.float()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

# Save the model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
