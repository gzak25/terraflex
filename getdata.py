import serial
import time
import pandas as pd
import os

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
DATA_DIR = './pressure_data_collection/'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

STATE = ['back']   # change label based on what data you're collecting
DURATION = .5      # Duration to collect data (in seconds)
REPETITIONS = 20   # Number of repetitions

def collect_pressure_data(state_label, repetition, duration=DURATION):
    print("Connecting to Arduino. Ensure bump switch is activated for data collection.")
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(2)  # Allow Arduino to reset
            print(f"Recording '{state_label}' (Rep: {repetition}) for {duration} second(s)...")
            start_time = time.time()
            data = []
            prefix = "Pressure Sensor Data "
            while time.time() - start_time < duration:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if not line:
                        continue
                    # Skip lines where bump switch is not activated
                    if "Bump switch not activated" in line:
                        continue
                    # Process lines that start with the sensor data header
                    if line.startswith(prefix):
                        sensor_data_str = line[len(prefix):]  # Extract numbers after the header
                        values = sensor_data_str.split(',')
                        if len(values) == 3:
                            try:
                                sensor_A0 = int(values[0])
                                sensor_A1 = int(values[1])
                                sensor_A2 = int(values[2])
                            except ValueError:
                                print(f"Invalid sensor values: {values}")
                                continue

                            timestamp = time.time()
                            data.append([timestamp, sensor_A0, sensor_A1, sensor_A2])
                        else:
                            print(f"Unexpected sensor data format: {sensor_data_str}")
                    else:
                        continue

                except Exception as e:
                    print(f"Error reading data: {e}")
                    break

            if data:
                df = pd.DataFrame(data, columns=['timestamp', 'pressure_sensor_A0', 'pressure_sensor_A1', 'pressure_sensor_A2'])
                file_path = os.path.join(DATA_DIR, f"{state_label}_rep{repetition}_{int(start_time)}.csv")
                df.to_csv(file_path, index=False)
                print(f"Data for '{state_label}' (Rep: {repetition}) saved to {file_path}.")
            else:
                print(f"No valid data collected for '{state_label}' (Rep: {repetition}).")

    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    print("Starting pressure sensor data collection...")
    for state in STATE:
        for rep in range(1, REPETITIONS + 1):
            input(f"(Rep {rep}/{REPETITIONS}): Ensure the bump switch is activated then press Enter to record '{state}' data...")
            collect_pressure_data(state_label=state, repetition=rep, duration=DURATION)
    print("Data collection complete.")
