"""Script to collect EMG and pressure sensor data from Arduino."""

import os
import time
import serial
import pandas as pd

SERIAL_PORT = "/dev/cu.usbserial-110"
BAUD_RATE = 9600
DATA_DIR = "./EMG_and_pressure_data/"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

STATE = ["stable"]  # change label based on what data you're collecting
DURATION = 0.5  # Duration to collect data (in seconds)
REPETITIONS = 20  # Number of repetitions


def collect_pressure_data(state_label, repetition, duration=DURATION):
    """
    Collects pressure sensor data from Arduino and saves it to a CSV file.
    """
    print("Connecting to Arduino.")
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(2)  # Allow Arduino to reset
            print(
                f"Recording '{state_label}' (Rep: {repetition}) for {duration} second(s)..."
            )
            start_time = time.time()
            data = []
            prefix = ""
            while time.time() - start_time < duration:
                try:
                    line = ser.readline().decode("utf-8").strip()
                    if not line:
                        continue

                    # Process lines that start with the sensor data header
                    sensor_data_str = line[
                        len(prefix) :
                    ]  # Extract numbers after the header
                    values = sensor_data_str.split(",")
                    if len(values) == 7:
                        try:
                            emg = int(values[0])
                            front_left_1 = int(values[1])
                            front_left_2 = int(values[2])
                            front_right_1 = int(values[3])
                            front_right_2 = int(values[4])
                            back_left = int(values[5])
                            back_right = int(values[6])
                        except ValueError:
                            print(f"Invalid sensor values: {values}")
                            continue

                        timestamp = time.time()
                        data.append(
                            [
                                timestamp,
                                emg,
                                front_left_1,
                                front_left_2,
                                front_right_1,
                                front_right_2,
                                back_left,
                                back_right,
                            ]
                        )
                    else:
                        print(f"Unexpected sensor data format: {sensor_data_str}")

                except Exception as e:
                    print(f"Error reading data: {e}")
                    break

            if data:
                df = pd.DataFrame(
                    data,
                    columns=[
                        "timestamp",
                        "emg",
                        "front_left_sensor_1",
                        "front_left_sensor_2",
                        "front_right_sensor_1",
                        "front_right_sensor_2",
                        "back_left_sensor",
                        "back_right_sensor",
                    ],
                )
                file_path = os.path.join(
                    DATA_DIR, f"{state_label}_rep{repetition}_{int(start_time)}.csv"
                )
                df.to_csv(file_path, index=False)
                print(
                    f"Data for '{state_label}' (Rep: {repetition}) saved to {file_path}."
                )
            else:
                print(
                    f"No valid data collected for '{state_label}' (Rep: {repetition})."
                )

    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    print("Starting pressure sensor data collection...")
    for state in STATE:
        for rep in range(1, REPETITIONS + 1):
            collect_pressure_data(state_label=state, repetition=rep, duration=DURATION)
    print("Data collection complete.")
