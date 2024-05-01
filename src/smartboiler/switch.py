# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# This module is used for turning on and off the Shelly devices.


import requests


class Switch:
    """Class for communicating with Shelly devices and turning them on or off."""

    def __init__(self, entity_id: str, base_url: str, token: str, headers: dict):
        """Init method for the Switch class

        Args:
            entity_id (str): Entity ID of the switch.
            base_url (str): Base URL of the Shelly device.
            token (str): Token for the Shelly device.
            headers (dict): Headers for the Shelly device.
        """
        self.entity_id = entity_id
        self.base_url = base_url
        self.token = token
        self.headers = headers

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
            endpoint = f"http://{self.base_url}/relay/0?turn={action}"

            requests.post(endpoint, timeout=5)

        except Exception as e:
            print(f"Failed to turn {action} Shelly with exception: {e}")
            pass
