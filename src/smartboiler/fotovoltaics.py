import requests


class Fotovoltaics:
    def __init__(
        self,
        power: float,
        efficiency: float,
        token: str,
        sn: str,
        battery_capacity: float,
        battery_power: float,
    ):
        self.power = power
        self.efficiency = efficiency
        self.token = token
        self.sn = sn
        self.stats_url = f"https://www.solaxcloud.com/proxyApp/proxy/api/getRealtimeInfo.do?tokenId={token}&sn={sn}"

    def get_actual_stats(self):
        # get actual stats from the fotovoltaics
        response = requests.get(self.stats_url)
        data = response.json()
        return data

    def get_actual_power(self):
        # get actual power from the fotovoltaics
        data = self.get_actual_stats()
        return data["result"]["acpower"]


    def get_actual_fve_production(self):
        # get actual production from the fotovoltaics
        data = self.get_actual_stats()
        return data["result"]["feedinpower"]

    def get_battery_level(self):
        # get battery level from the fotovoltaics
        data = self.get_actual_stats()
        return data["result"]["soc"]

    def is_battery_charging(self) -> bool:
        # check if the battery is charging
        return True if self.get_battery_level() < 100 else False

    def is_consumption_lower_than_production(self) -> bool:
        # check if the consumption is lower than production
        return (
            True
            if self.get_actual_power() < self.get_actual_fve_production()
            else False
        )
        
        
if __name__ == "__main__":
    fotovoltaics = Fotovoltaics(
        power=100,
        efficiency=0.9,
        token="20240329202534682538500",
        sn="SRDHKJ3HJC",
        battery_capacity=10,
        battery_power=5)
    
    print(fotovoltaics.get_actual_stats())
    print(fotovoltaics.is_battery_charging())
    print(fotovoltaics.is_consumption_lower_than_production())