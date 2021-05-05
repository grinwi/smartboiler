This is a manual for creating your own smart boiler from a dumb one.

# Get the accessories
The first thing you need to do is to get a smart socket to which will be connected boiler and a device, which will be able to measure temperature from a body of boiler. 
The best and the easiest option is to buy a smart relay Shelly 1PM with a module for connecting two temperature sensors DS18B20. On this relay connect the socket to which is connected the boiler.

![Connection of smart relay on a socket.](https://github.com/grinwi/smart_boiler/blob/main/relay.png)

# Connect sensors
The two temperature sensors need to be situated on a boiler. First one place into a case of boiler close to the top of the bottle tank. This one will be monitoring the temperature of the water in the boiler. Second one place on an output pipe of hot water for detecting the consumption.

# Set up the socket
The next thing, which is needed to do, is to prepare the devices for API communication on a local network under a static IP. If using the Shelly 1PM, set this via the web interface of a socket by connecting to a Wi-Fi network of the relay and connect to the same local network where will be connected the PC to control the relay. Then set the static IP of a relay and this IP write into a configure JSON file settings.json together with parameters of a boiler etc.

If you decide to use other devices for controlling the boiler and measuring the temperature, you have to change the source codes in control.py and sensor.py!

# Download the Google Calendar Credentials
Following the manual at https://developers.google.com/calendar/quickstart/python create the credentials for connecting the API of the Google Calendar, where you can place the events for controlling the boiler. These credentials place under the same filenames into a folder "control". 

You can use the preset events as:
- Holiday
Place a string "#off" into a name of an event and during this event, the water in the boiler won't be heated.
- Prepare shower
Using the string "Prepare <number of showers> showers" in the name of an event assure the required number of showers, when as one shower is 40 liters water of 40 degrees Celsium.
- Prepare water at a certain temperature
The event with string "boiler heat up at <number> degrees" in the name of an event ensures the water of required temperature prepared on a start of the event."

# Set up the workstation
For a run of an algorithm of a smart boiler, it is needed to run the script in a Docker. For this, it's necessary to connect the containers to your own Docker network.  The script is then easily run by a docker-compose command. If you want to see the stats of boiler as a temperature or consumption of electricity, use the interface of the Grafana, where you can select in which measurements and values are you interested
How to manage the Docker containers including creating the network is described for example on this link: <https://thedatafrog.com/en/articles/docker-influxdb-grafana/>

# Enjoy the advantages of a smart boiler!
