// Data collection of the EMG and Pressure sensors 

#include <Servo.h>

void setup() {
  Serial.begin(9600);
}

// Main loop to read and print sensor data
void loop() {
  readSensors();
  serialPrintData();
  delay(500);
}

// Function to read the EMG and Pressure sensor values
void readSensors() {
  int emgValue = analogRead(A1);
  int frontLeft1Value = analogRead(A2); // Pressure sensor in front left toe
  int frontLeft2Value = analogRead(A3); // Pressure sensor in front left middle
  int frontRight1Value = analogRead(A4); // Pressure sensor in front right toe
  int frontRight2Value = analogRead(A5); // Pressure sensor in front right middle
  int backLeftValue = analogRead(A6); // Pressure sensor in back left toe
  int backRightValue = analogRead(A7); // Pressure sensor in back right toe
}

// Function to print the data to the serial monitor
void serialPrintData() {
  Serial.print(emgValue);
  Serial.print(',');
  Serial.print(pressureA2);
  Serial.print(',');
  Serial.print(pressureA3);
  Serial.print(',');
  Serial.print(pressureA4);
  Serial.print(',');
  Serial.print(pressureA5);
  Serial.print(',');
  Serial.print(pressureA6);
  Serial.print(',');
  Serial.println(pressureA7);
  
}
