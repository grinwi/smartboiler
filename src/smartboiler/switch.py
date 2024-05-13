# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# This module is used for turning on and off the Shelly devices.


import requests


class Switch:
    """Class for communicating with Shelly devices and turning them on or off."""

    def __init__(self, shelly_ip: str):
        """Init method for the Switch class

        Args:
            shelly_ip (str): IP address of the Shelly device.
        """
        self.shelly_ip = shelly_ip

    def turn_on(self):
        """Turns the shelly on."""
        self._turn_action("on")

    def turn_off(self):
        """Turns the shelly off."""
        self._turn_action("off")

    def _turn_action(self, action: str):
        """Try to turn the Shelly on or off.

        Args:
            action (str): String to turn the Shelly on or off.
        """
        try:
            endpoint = f"http://{self.shelly_ip}/relay/0?turn={action}"

            requests.post(endpoint, timeout=5)

        except Exception as e:
            print(f"Failed to turn {action} Shelly with exception: {e}")
            pass
