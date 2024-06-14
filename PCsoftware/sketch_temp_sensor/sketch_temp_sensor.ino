#include <dht_nonblocking.h>
#define DHT_SENSOR_TYPE DHT_TYPE_11

static const int DHT_SENSOR_PIN = 3;
int low_temp = 8;
int high_temp = 9;
int high_humi = 10;

DHT_nonblocking dht_sensor( DHT_SENSOR_PIN, DHT_SENSOR_TYPE );


/*
 * Initialize the serial port.
 */
void setup( )
{
  Serial.begin( 9600);
  pinMode(low_temp, OUTPUT);
  pinMode(high_temp, OUTPUT);
  pinMode(high_humi, OUTPUT);

}



/*
 * Poll for a measurement, keeping the state machine alive.  Returns
 * true if a measurement is available.
 */
static bool measure_environment( float *temperature, float *humidity )
{
  static unsigned long measurement_timestamp = millis( );

  /* Measure once every four seconds. */
  if( millis( ) - measurement_timestamp > 3000ul )
  {
    if( dht_sensor.measure( temperature, humidity ) == true )
    {
      measurement_timestamp = millis( );
      return( true );
    }
  }

  return( false );
}



/*
 * Main program loop.
 */
void loop( )
{
  float temperature;
  float humidity;

  /* Measure temperature and humidity.  If the functions returns
     true, then a measurement is available. */
  if( measure_environment( &temperature, &humidity ) == true )
  {
    Serial.print( "T = " );
    Serial.print( temperature, 1 );
    Serial.print( " deg. C, H = " );
    Serial.print( humidity, 1 );
    Serial.println( "%" );
    if(temperature<18.0){
      digitalWrite(low_temp, HIGH);
    }
    else {digitalWrite(low_temp, LOW);}
    if(temperature>25.0){
      digitalWrite(high_temp, HIGH);
    }
    else {digitalWrite(high_temp, LOW);}
    if(humidity>75.0){
      digitalWrite(high_humi, HIGH);
    }
    else {digitalWrite(high_humi, LOW);}

  }
}