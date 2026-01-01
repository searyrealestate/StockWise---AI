/*
  # Example code for the moisture sensor
  # Editor     : Lauren
  # Date       : 13.01.2012
  # Version    : 1.0
  # Connect the sensor to the A0(Analog 0) pin on the Arduino board

  # the sensor value description
  # 0  ~300     dry soil
  # 300~700     humid soil
  # 700~950     in water
*/

void setup(){

  Serial.begin(57600);
  Serial.println("Here we go:");

}
#define N_AVERAGE 32
int counter = 0;

const int AirValue = 520;   //you need to replace this value with Value_1
const int WaterValue = 250;  //you need to replace this value with Value_2
int intervals = (AirValue - WaterValue)/3;
int soilMoistureValue = 0;

long int sum1= 0, sum2 = 0, sum3 = 0;

void loop(){
  int i; 
  sum1 = 0;
  sum2 = 0;
  sum3 = 0;
  for (i=0;i<N_AVERAGE;i++){
    sum1 += analogRead(A0);
    sum2 += analogRead(A1);
    sum3 += analogRead(A2);
    delay(50);
  }

  soilMoistureValue = sum2/N_AVERAGE;

  Serial.print( counter);
  Serial.print(": Moisture Sensor Value:");
  Serial.print(sum1/N_AVERAGE);
  Serial.print(": Moisture Sensor 2 Value:");
  Serial.print(sum3/N_AVERAGE);
  Serial.print(", Capacitive Moisture Sensor Value:");
  Serial.println(soilMoistureValue);
  if(soilMoistureValue > WaterValue && soilMoistureValue < (WaterValue + intervals))
  {
    Serial.println("Very Wet");
  }
  else if(soilMoistureValue > (WaterValue + intervals) && soilMoistureValue < (AirValue - intervals))
  {
    Serial.println("Wet");
  }
  else if(soilMoistureValue < AirValue && soilMoistureValue > (AirValue - intervals))
  {
    Serial.println("Dry");
  }

  
  counter++;
}