from flask import Flask, request, jsonify
from server_with_ML import OracleBotThread

app = Flask(__name__)

oracle_bot_thread = OracleBotThread()

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(force=True)
    intent_name = req.get('queryResult').get('intent').get('displayName')

    if intent_name == 'WeatherIntent':
        
        date = req.get('queryResult').get('parameters').get('date')
        
       
        weather_data = oracle_bot_thread.predict_weather(date)
        
        
        fulfillment_text = (f"For {date}, the forecast is: "
                            f"high of {weather_data['high_temp']:.1f}°C, "
                            f"low of {weather_data['low_temp']:.1f}°C, "
                            f"wind speed at {weather_data['wind_speed']:.1f} km/h, 
"
                            f"and precipitation levels around 
{weather_data['precip']:.1f} mm.")
    elif intent_name == 'BikeHirePrediction':
        
        day = req.get('queryResult').get('parameters').get('day')
        
        
        prediction_response = oracle_bot_thread.predict_bike_hires(day)
        
        
        fulfillment_text = f"For {day}, the expected number of bike hires is 
approximately {prediction_response}."
    else:
        fulfillment_text = "I'm sorry, I didn't understand. Ask about weather or 
bike hires."

    
    return jsonify({
        "fulfillmentMessages": [
            {
                "text": {
                    "text": [fulfillment_text]
                }
            }
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
