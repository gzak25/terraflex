// Arduino sketch to read pressure sensor values from A0, A1, and A2
// only when bump switch is activated

const int sensorPinA0 = A0;  // Pressure sensor 1
const int sensorPinA1 = A1;  // Pressure sensor 2
const int sensorPinA2 = A2;  // Pressure sensor 3
const int bumpSwitchPin = 8; // Bump switch

void setup() {
  Serial.begin(9600);
  pinMode(bumpSwitchPin, INPUT);
}

void loop() {
  int bumpState = digitalRead(bumpSwitchPin);

  if (bumpState == LOW) { // Bump switch is activated
    int pressureA0 = analogRead(sensorPinA0);
    int pressureA1 = analogRead(sensorPinA1);
    int pressureA2 = analogRead(sensorPinA2);

    Serial.print("Pressure Sensor Data ");
    Serial.print(pressureA0);
    Serial.print(',');
    Serial.print(pressureA1);
    Serial.print(',');
    Serial.println(pressureA2);
  } else {
    Serial.println("Bump switch not activated. Sensors inactive.");
  }

  delay(5); 
}
