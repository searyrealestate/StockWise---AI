// Motor A connections
int enA = 6;
int in1 = 2;
int in2 = 3;
// Motor B connections
int enB = 11;
int in3 = 4;
int in4 = 5;

int buzzer = 12; 

int triggerPin = 9;
int echoPin = 10;
int speed = 255;
int distance = 20;

long readUltrasonicDistance(int triggerPin, int echoPin)
{
  pinMode(triggerPin, OUTPUT);  // Clear the trigger
  digitalWrite(triggerPin, LOW);
  delayMicroseconds(2);
  // Sets the trigger pin to HIGH state for 10 microseconds
  digitalWrite(triggerPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(triggerPin, LOW);
  pinMode(echoPin, INPUT);
  // Reads the echo pin, and returns the sound wave travel time in microseconds
  return pulseIn(echoPin, HIGH);
}

void setup() {
  // Set all the motor control pins to outputs
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  
  // Turn off motors - Initial state
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

   // ultrasonic
  pinMode(triggerPin, OUTPUT); 
  pinMode(echoPin, INPUT);
  Serial.begin(9600);

  pinMode(buzzer, OUTPUT);

  analogWrite(enA, 200);
  analogWrite(enB, 200);

   // ultrasonic
  pinMode(triggerPin, OUTPUT); 
  pinMode(echoPin, INPUT);
  Serial.begin(9600);

  pinMode(buzzer, OUTPUT);


}
void loop() {

  if (check_distance()){
    stop();
    delay(5);
    drive_back();
    buzzer_on();
    delay(500);
    drive_right();
    delay(100);
    //stop();
  } else {
    drive_forword();
    buzzer_off();
    delay(10);
  }
}

void stop(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}
//drive forword
void drive_forword(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

//drive back
void drive_back(){
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}

//drive right
void drive_right(){
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

//drive left
void drive_left(){
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}

// buzzer on
void buzzer_on(){
  digitalWrite(buzzer, HIGH);
}

// buzzer off
void buzzer_off(){
  digitalWrite(buzzer, LOW);
}

// ultrasonic - check bellow minimal distance
bool check_distance(){
  if (0.01723 * readUltrasonicDistance(triggerPin, echoPin) < distance) {
    return true;
  } else{
    return false;
  }
// This function lets you control speed of the motors

}
