# smart_bojler
From dumb boiler smart with help of smart plug. 


## What's it all about?
This project is my bachelor's thesis with one simple goal: make from dumb boiler a smart one with help of smart plug and few sensors.
The smart part is, that the heating of boiler is based on historic data of consumption from boiler. 
That provides, that boiler doesnt have to be heated up at max temperature all the time, so the heat loses will be smaller.
There is also possibility to link the solution with an Google Calendar. 
In a calendar can be set an events for holidays, when the boiler won't heat, or events describing an unordinary consumption following with temperature, on which should bethe boiler heated on.


## How does it work?
The solution is made with SHELLY 1PM plug and an additional temperature modul with two connected DS18B20's. One is inside of a boiler and the second one on an output pipe of a boiler.
Testing and appliance is managed by docker containers. One for collecting data from a plug, second for controlling of the plug. There is also one more for an InfluxDB where are datas collected from plug stored.
Review is managed by graphana container connected to InfluxDB.
### Problems in solution
Hardest part was find out, how mch of water was consumed based on temperature of an output pipe. 
This is managed by averaging a temperature during a days, searching for a peaks in consumption and selecting possible consumptions.
The peak's top temperature is multiplied by a coeficient, which is edited with regard to maximal reducing of an average temperature in boiler but also due toheat comfort of a water.

The second hardest thing was to guess or estimate a real temperature of watter in boiler, because this thermometer couldn't be placed right into a water tank but only in an isolation, 
so the temperature on a sensor is lower than it actually is and there is also and extrapolation in time, when the water is heated on.
