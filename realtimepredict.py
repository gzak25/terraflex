import torch
import numpy as np
import serial
import torch.nn as nn
import torch.nn.functional as F
import time

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

window_size = 20  # Adjust based on how often you want predictions to happen


class CNN(nn.Module):
    def __init__(self, num_classes):

        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x


model_path = "model.pth"
num_classes = 5  # Categories: stable, forward, right, left, back
model_cnn = CNN(num_classes=num_classes)
model_cnn.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model_cnn.eval()

# Balance mapping
balance_mapping = {0: "stable", 1: "forward", 2: "right", 3: "left", 4: "back"}


def parse_sensor_line(line):
    """
    Parses a line of serial data expected in the format:
    "Pressure Sensor Data <valA0>,<valA1>,<valA2>"
    Returns a list of three integer sensor readings.
    """
    prefix = "Pressure Sensor Data "
    if not line.startswith(prefix):
        return None
    try:
        data_str = line[len(prefix) :].strip()
        values = data_str.split(",")
        if len(values) != 3:
            return None
        return [int(v) for v in values]
    except ValueError:
        return None


def predict_from_buffer(buffer):
    """
    Converts a buffer of sensor readings into a tensor, performs prediction,
    and returns the predicted balance label.
    """
    # Convert buffer (list of lists) to numpy array of shape (window_size, 3)
    data_np = np.array(buffer)
    # Transpose to get shape (3, window_size)
    data_np = data_np.T
    sensor_tensor = torch.FloatTensor(data_np).unsqueeze(
        0
    )  # Shape: (1, 3, window_size)
    with torch.no_grad():
        output = model_cnn(sensor_tensor)
        _, predicted = torch.max(output, 1)
    balance_idx = predicted.item()
    return balance_mapping.get(balance_idx, "Unknown")


def send_to_arduino(gesture_name):
    """
    Sends the predicted gesture to the Arduino over the serial port.
    """
    ser.write(f"{gesture_name}\n".encode("utf-8"))
    time.sleep(0.1)
    while ser.in_waiting > 0:
        response = ser.readline().decode("utf-8").strip()
        print(f"Arduino response: {response}")


if __name__ == "__main__":
    print("Starting live balance prediction. Press Ctrl+C to exit.")
    sensor_buffer = []
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            # Only process valid sensor data lines.
            sensor_values = parse_sensor_line(line)
            if sensor_values is not None:
                sensor_buffer.append(sensor_values)
                # Predict balance when we have enough data
                if len(sensor_buffer) >= window_size:
                    predicted_label = predict_from_buffer(sensor_buffer[:window_size])
                    print(f"Predicted: {predicted_label}")
                    send_to_arduino(predicted_label)
                    sensor_buffer = []

            else:
                continue
    except KeyboardInterrupt:
        print("Exiting live prediction.")
        ser.close()
