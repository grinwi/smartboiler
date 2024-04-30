
# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# This module is used for communicating with Solax fotovoltaics and retrieving the actual stats.


import requests

class Fotovoltaics:
    """Class for communicating with solaxcloud api and retrieving the actual stats.
    """
    def __init__(
        self,
        power: float,
        efficiency: float,
        token: str,
        sn: str,
    ):
        """Initialize the class with the power, efficiency, token and sn of the Solax fotovoltaics.

        Args:
            power (float): Power of the fotovoltaics on W.
            efficiency (float): Efficiency of the fotovoltaics (0,1).
            token (str): Token for the Solax API.
            sn (str): SN of the Solax fotovoltaics.
        """
        self.power = power
        self.efficiency = efficiency
        self.token = token
        self.sn = sn
        self.stats_url = f"https://www.solaxcloud.com/proxyApp/proxy/api/getRealtimeInfo.do?tokenId={token}&sn={sn}"

    def get_actual_stats(self) -> dict:
        """Method calling the solax API and returning the actual stats.

        Returns:
            dict: Dictionary with the response from the API.
        """
        # get actual stats from the fotovoltaics
        response = requests.get(self.stats_url)
        data = response.json()
        return data

    def get_actual_power(self) -> float:
        """Method to get actual power from the fotovoltaics.

        Returns:
            float: AC power
        """
        data = self.get_actual_stats()
        return data["result"]["acpower"]

    def get_actual_fve_production(self) -> float:
        """Method to get actual fotovoltaics production.

        Returns:
            float: Feeding power.
        """
        data = self.get_actual_stats()
        return data["result"]["feedinpower"]

    def get_battery_level(self) -> float:
        """Method to get the battery level.

        Returns:
            float: Percentage of the battery level.
        """
        data = self.get_actual_stats()
        return data["result"]["soc"]

    def is_battery_charging(self) -> bool:
        """Method to check if the battery is charging.

        Returns:
            bool: Information abour charging status of the battery, True if charging, False if not
        """
        return True if self.get_battery_level() < 100 else False

    def is_consumption_lower_than_production(self) -> bool:
        """Method to check if the consumption is lower than the production.

        Returns:
            bool: is_consumption_lower_than_production.
        """
        return (
            True
            if self.get_actual_power() < self.get_actual_fve_production()
            else False
        )


if __name__ == "__main__":
    fotovoltaics = Fotovoltaics(
        power=100,
        efficiency=0.9,
        token="",
        sn="",
        battery_capacity=10,
        battery_power=5,
    )

    print(fotovoltaics.get_actual_stats())
    print(fotovoltaics.is_battery_charging())
    print(fotovoltaics.is_consumption_lower_than_production())
