// Data collection of the EMG and Pressure sensors 
#include <Servo.h>

// Global sensor variables
int emgValue;
int frontLeft1Value;
int frontLeft2Value;
int frontRight1Value;
int frontRight2Value;
int backLeftValue;
int backRightValue;

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
  emgValue = analogRead(A1);
  frontLeft1Value = analogRead(A2); // Pressure sensor in front left toe
  frontLeft2Value = analogRead(A3); // Pressure sensor in front left middle
  frontRight1Value = analogRead(A4); // Pressure sensor in front right toe
  frontRight2Value = analogRead(A5); // Pressure sensor in front right middle
  backLeftValue = analogRead(A6); // Pressure sensor in back left toe
  backRightValue = analogRead(A7); // Pressure sensor in back right toe
}

// Function to print the data to the serial monitor
void serialPrintData() {
  Serial.print(emgValue); Serial.print(',');
  Serial.print(frontLeft1Value); Serial.print(',');
  Serial.print(frontLeft2Value); Serial.print(',');
  Serial.print(frontRight1Value); Serial.print(',');
  Serial.print(frontRight1Value); Serial.print(',');
  Serial.print(backLeftValue); Serial.print(',');
  Serial.println(backRightValue);
}
