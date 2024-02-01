import requests

class Switch:
    def __init__(self, entity_id, url, token, headers):
        self.entity_id = entity_id
        self.url = url
        self.token = token
        self.headers = headers


    def turn_on(self):
        if self.entity_id == 'switch.smartboiler':
            data = {"entity_id": self.entity_id, "brightness": 255}
            response = requests.post(self.url + '/api/services/light/turn_on', json=data, headers=self.headers)
            print(response)
