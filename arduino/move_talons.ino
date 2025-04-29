// This code is used for controlling the talons depending the predicted label

#include <Servo.h>

// Define servo objects
Servo frontLeftServo;
Servo frontRightServo;
Servo backLeftServo;
Servo backRightServo;

// Pins
const int frontLeftPin = 7; 
const int frontRightPin = 5;
const int backLeftPin = 8;
const int backRightPin = 6;

// Servo positions
const int OPEN_POS = 0;
const int CLOSED_POS = 180;

// Variables that were previously `const`
String status = "stable";
bool moveInProgress = false;

// Sample rate timing
unsigned long sampleRate = 100;
unsigned long previousMillis = 0;

// Global sensor variables
int emgValue;
int frontLeft1Value;
int frontLeft2Value;
int frontRight1Value;
int frontRight2Value;
int backLeftValue;
int backRightValue;

void setup() {
  frontLeftServo.attach(frontLeftPin);
  frontRightServo.attach(frontRightPin);
  backLeftServo.attach(backLeftPin);
  backRightServo.attach(backRightPin);

  frontLeftServo.write(OPEN_POS);
  frontRightServo.write(OPEN_POS);
  backLeftServo.write(OPEN_POS);
  backRightServo.write(OPEN_POS);

  Serial.begin(9600);
}

void loop() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= (1000 / sampleRate)) {
    previousMillis = currentMillis;
    readSensors();
    serialPrintData();
  }

  if (Serial.available() > 0 && !moveInProgress) {
    String input = Serial.readStringUntil('\n');
    // Serial.print("got input----");
    // Serial.print(input);
    // Serial.print("-----");
    input.trim();
    // Serial.println(input);
    
    if (input.length() > 0 && input != status) {
      Serial.print("Got input: ");
      Serial.println(input);
      status = input;
      moveTalons();
    }
  }
}

void moveTalons() {
  Serial.println("-----Moving talons------");
  moveInProgress = true;
  if (status == "stable") {
    stable();
  } else if (status == "forward") {
    forward();
  } else if (status == "backward") {
    back();
  } else if (status == "left") {
    left();
  } else if (status == "right") {
    right();
  } else {
    Serial.println("The status should be one of the stable/forward/backward/left/right.");
  }
  moveInProgress = false;
}

void back() {
  Serial.println("Back movement");
  frontLeftServo.write(CLOSED_POS);
  frontRightServo.write(CLOSED_POS);
  backLeftServo.write(OPEN_POS);
  backRightServo.write(OPEN_POS);
}

void forward() {
  Serial.println("Forward movement");
  frontLeftServo.write(OPEN_POS);
  frontRightServo.write(OPEN_POS);
  backLeftServo.write(CLOSED_POS);
  backRightServo.write(CLOSED_POS);
}

void left() {
  Serial.println("Left movement");
  frontLeftServo.write(OPEN_POS);
  frontRightServo.write(CLOSED_POS);
  backLeftServo.write(OPEN_POS);
  backRightServo.write(CLOSED_POS);
}

void right() {
  Serial.println("Right movement");
  frontLeftServo.write(CLOSED_POS);
  frontRightServo.write(OPEN_POS);
  backLeftServo.write(CLOSED_POS);
  backRightServo.write(OPEN_POS);
}

void stable() {
  Serial.println("Already stable");
  frontLeftServo.write(OPEN_POS);
  frontRightServo.write(OPEN_POS);
  backLeftServo.write(OPEN_POS);
  backRightServo.write(OPEN_POS);
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
