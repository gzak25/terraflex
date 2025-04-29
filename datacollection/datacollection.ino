#include <Servo.h>

// Code tested for second sprint review.
// Can move motors based on rock, paper, scissors movement

// Define servo objects
Servo frontLeftServo;
Servo frontRightServo;
Servo backLeftServo;
Servo backRightServo;

// Places where servos are connected
const int frontLeftPin = 7; 
const int frontRightPin = 5;
const int backLeftPin = 8;
const int backRightPin = 6;
const int THRESHOLD = 150; 
const int OPEN_POS = 0;     // Servo open position
const int CLOSED_POS = 180;  // Servos closed position

const int sensorPinA2 = A2;  // Pressure sensor 1
const int sensorPinA3 = A3;  // Pressure sensor 2
const int sensorPinA4 = A4;  // Pressure sensor 3
const int sensorPinA5 = A5;  // Pressure sensor 1
const int sensorPinA6 = A6;  // Pressure sensor 2
const int sensorPinA7 = A7;  // Pressure sensor 3

void setup() {
  // Attach servos to their ports
  frontLeftServo.attach(frontLeftPin);
  frontRightServo.attach(frontRightPin);
  backLeftServo.attach(backLeftPin);
  backRightServo.attach(backRightPin);
  pinkyServo.attach(pinkyPin);

  // Set all servos to open
  frontLeftServo.write(OPEN_POS);
  frontRightServo.write(OPEN_POS);
  backLeftServo.write(OPEN_POS);
  backRightServo.write(OPEN_POS);
  pinkyServo.write(OPEN_POS);

  Serial.begin(9600);
}

void loop() {
  int pressureA2 = analogRead(sensorPinA2);
  int pressureA3 = analogRead(sensorPinA3);
  int pressureA4 = analogRead(sensorPinA4);
  int pressureA5 = analogRead(sensorPinA5);
  int pressureA6 = analogRead(sensorPinA6);
  int pressureA7 = analogRead(sensorPinA7);
  int emgValue = analogRead(A0);
  collectEMGData();
  collectPressureData();
  delay(500);
}

// Function to show Paper gesture
void collectPressureData() {
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

void collectEMGData(){
  Serial.print(emgValue);
  Serial.print(',');
}
