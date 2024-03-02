import requests

class Switch:
    def __init__(self, entity_id, base_url, token, headers):
        self.entity_id = entity_id
        self.base_url = base_url
        self.token = token
        self.headers = headers


    def turn_on(self):
        self._turn_action("on")
            
    def turn_off(self):
        self._turn_action("off")
            
    def _turn_action(self, action):
        data = {'entity_id': self.entity_id}
        print("Setting shelly relay to OFF")
        response = requests.post(
            f"{self.base_url}/services/switch/{action}", headers=self.headers, json=data
        )
        if response.status_code == 200:
            print(f"Shelly turned {action} successfully")
        else:
            print("Failed to turn {action} Shelly")