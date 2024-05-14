
# Set up the Home Assistant
Create a Home Assistant instance in your home. Install InfluxDB addon and run your first database, where will be the data from smarthome stored.
Set up also Nmap Tracker integration to scan your network for number of the connected devices.
The AccuWeather Add-On should be also set to retrieve the actual weather info used for creating of the prediction.

# Get the accessories
The first thing you need to do is to get a smart socket to which will be connected boiler and a device, which will be able to measure temperature from a body of boiler. 
The best and the easiest option is to buy a smart relay Shelly 1PM with a module for connecting temperature sensors DS18B20. On this relay connect the socket to which is connected the boiler.
Second one physically needed is flow meter to calculate the amount of heat consumed. You can use YF-B6 with a temperature sensor or any other which can be integrated into a Home Assistant via microcotroller with ESPHome or any other solution. 

# Connect sensors
Temperature sensor need to be situated on a boiler. Place it into a case of boiler close to the top of the bottle tank. This one will be monitoring the temperature of the water in the boiler. 
Second temperature sensor should be part of the flow meter on the output of the heat water from the boiler. You can use YF-B6 with temperature sensor connected to Wemos D1 mini with ESPHome flashed and integrated into a Home Assistant

# Set up the socket
The next thing, which is needed to do, is to prepare the devices for API communication on a local network under a static IP. If using the Shelly 1PM, set this via the web interface of a socket by connecting to a Wi-Fi network of the relay and connect to the same local network where will be connected the Home Assistant to control the relay. 
The IP of the socket will be inserted in Home Assitant Smart Boiler Add-On configuration page.

If you decide to use other devices for controlling the boiler and measuring the temperature, you have to change the source codes in controller.py.

# Download the Google Calendar Credentials
Following the manual at https://developers.google.com/calendar/quickstart/python create the credentials for connecting the API of the Google Calendar, where you can place the events for controlling the boiler. Then generate the token.json, which you will place with the same name into a /app sensor inside of the Smart Boiler Add-On container. To achieve this, you can simply use a Portainer add-on and echo TEXT_OF_THE_TOKEN_FILE > token.json .

You can use the preset events as:
- Holiday
Place a string "#off" into a name of an event and during this event, the water in the boiler won't be heated.
- Prepare water at a certain temperature
The event with string "heat water at <number> degrees" in the name of an event ensures the water of required temperature prepared on a start of the event."


# Enjoy the advantages of a water heating with use of the machone learning!
