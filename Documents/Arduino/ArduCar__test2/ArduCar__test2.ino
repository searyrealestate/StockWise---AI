// Motor A connections
int enA = 6;
int in1 = 2;
int in2 = 3;
// Motor B connections
int enB = 11;
int in3 = 4;
int in4 = 5;

int buzzer = 12; 

int trigPin = 9;
int echoPin = 10;


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
  pinMode(trigPin, OUTPUT); 
  pinMode(echoPin, INPUT);
  Serial.begin(9600);

  pinMode(buzzer, OUTPUT);

  analogWrite(enA, 200);
  analogWrite(enB, 200);

}
void loop() {
  // Speed - A
  // analogWrite(enA, 1);
  // analogWrite(enB, 1);
  // motor on
  // Stop Motor
  // Turn off motor A & B
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

  // drive back
  //digitalWrite(in1, HIGH);
 // digitalWrite(in2, LOW);
 // digitalWrite(in3, LOW);
 // digitalWrite(in4, HIGH);



  // drive right
  // Turn off motor A & B
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);

// buzzer on
  //digitalWrite(buzzer, HIGH);




  // drive left
  //digitalWrite(in1, LOW);
  //digitalWrite(in2, HIGH);
  //digitalWrite(in3, HIGH);
  //digitalWrite(in4, LOW);

  // drive back
  //digitalWrite(in1, LOW);
  //digitalWrite(in2, HIGH);
  //digitalWrite(in3, HIGH);
  //digitalWrite(in4, LOW);
  delay(2000);

// buzzer off
  //digitalWrite(buzzer, LOW);
  //delay(2000);



  // Speed - B - STOP
  // analogWrite(enA, 100);
  //analogWrite(enB, 100);
  // motor on
 // digitalWrite(in1, LOW);
 // digitalWrite(in2, LOW);
 // digitalWrite(in3, LOW);
 // digitalWrite(in4, LOW);
 // delay(1000);


// Speed - C
 //  analogWrite(enA, 200); 
  // analogWrite(enB, 200);
 // motor on
 // digitalWrite(in1, LOW);
 // digitalWrite(in2, HIGH);
 // digitalWrite(in3, LOW);
 // digitalWrite(in4, HIGH);
 // delay(2000);



  // Speed - D - STOP
 // analogWrite(enA, 100);
  //analogWrite(enB, 100);
 // motor on
 // digitalWrite(in1, LOW);
 // digitalWrite(in2, LOW);
 // digitalWrite(in3, LOW);
 // digitalWrite(in4, LOW);
 // delay(1000);
}
