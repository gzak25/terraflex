import time
import torch
import numpy as np
import serial
from cnn import CNN

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

WINDOW_SIZE = 20  # Adjust based on how often you want predictions to happen


MODEL_PATH = "model.pth"
NUM_CLASSES = 5  # Categories: stable, forward, right, left, back
model_cnn = CNN(num_classes=NUM_CLASSES)
model_cnn.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
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
                if len(sensor_buffer) >= WINDOW_SIZE:
                    predicted_label = predict_from_buffer(sensor_buffer[:WINDOW_SIZE])
                    print(f"Predicted: {predicted_label}")
                    send_to_arduino(predicted_label)
                    sensor_buffer = []
            else:
                continue
    except KeyboardInterrupt:
        print("Exiting live prediction.")
        ser.close()
