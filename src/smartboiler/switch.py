from urllib import response
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
        try:
            data = {"entity_id": self.entity_id}
            endpoint = f"http://{self.base_url}/relay/0?turn={action}"

            # print(f"Calling {endpoint} with data {data}")

            response = requests.post(endpoint, timeout=5)

        except Exception as e:
            print(f"Failed to turn {action} Shelly with exception: {e}")
            pass
