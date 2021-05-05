This is an manual for creating your own smart boiler from a dumb one.

# Get the accesories
First thing you need to do is to get a smart socket to which will be connected boiler and a device, which will be able to measure temperature from a body of boiler. 
The best and the easiest option is to buy a smart rele Shelly 1PM with a module for connect two temperature sensors DS18B20. On this rele connect the socket to which is connected the boiler.

![alt text](https://github.com/grinwi/smart_boiler/blob/main/sta%C5%BEen%C3%BD%20soubor.png)

# Connect sensors
The who temrperature sensors need to be situated on a boiler. First one place into a case of boiler. This one will be monitoring temperature of a water in boiler. Second one place on an output pipe for detecting the consumption.

# Set up the socket
Next thing, which is needed is to prepare the devices for an API communication on a local network under a static IP. This IP write into a configure JSON file settings.json together with parameters of a boiler etc.

# Download the Google Calendar Credentials
Following the manual at https://developers.google.com/calendar/quickstart/python create the credentials for connecting the Google Calendar API. This credentials place under the same filenames into a folder "control".

# Set up the workstation
For a run of a algorithm of a smart boiler it is needed to run the script in a Docker. For this, it's neccessary to connect the containers to your own Docker network.  The script is then easily run by a docker-compose command.

# Enjoy the advantages of a smart boiler!
