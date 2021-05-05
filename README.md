# SMART BOILER
From dumb boiler smart with help of smart plug. 

Recursive link to the github repository: <https://github.com/grinwi/smart_boiler>.


## What's it all about?
This project is my bachelor's thesis with one simple goal: make from dumb boiler a smart one with help of a smart plug and few sensors.
The smart part is, that the heating of the boiler is based on historic data of consumption from the boiler. 
That provides, that boiler doesn't have to be heated up at max temperature all the time, so the heat losses will be smaller.
There is also the possibility to link the solution with a Google Calendar. 
In a calendar can be set events for holidays, when the boiler won't heat, or events describing an unordinary consumption following with temperature, on which should be the boiler heated on.


## How does it work?
The solution is made with SHELLY 1PM plug and an additional temperature module with two connected DS18B20's. One is inside of a boiler and the second one on an output pipe of a boiler.
Testing and appliance are managed by docker containers. One for collecting data from a plug, the second for controlling the plug. There is also one more for an InfluxDB where are data collected from plug stored.
Review is managed by Graphana container connected to InfluxDB.
### Problems in solution
The hardest part was finding out, how much water was consumed based on a temperature of an output pipe. 
This is managed by averaging a temperature during days, searching for peaks in consumption, and selecting possible consumptions.
The peak's top temperature is multiplied by a coefficient, which is edited concerning maximal reducing of an average temperature in the boiler but also due to heat comfort of water.

The second hardest thing was to guess or estimate the real temperature of water in the boiler because this thermometer couldn't be placed right into a water tank but only in an isolation, 
so the temperature on a sensor is lower than it actually is and there is also an extrapolation in time when the water is heated on.
