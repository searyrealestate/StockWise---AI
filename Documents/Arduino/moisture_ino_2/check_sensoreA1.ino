// C++ code
//
void setup()
{
  pinMode(A1, INPUT);
  Serial.begin(9600);
}

void loop()
{
  Serial.print(analogRead(A1));
  delay(10); // Delay a little bit to improve simulation performance
}