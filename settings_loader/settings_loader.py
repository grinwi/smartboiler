import json
import os.path

class SettingsLoader:

    def __init__(self, file_name):
        if file_name is None:
            file_name = 'settings.json'
        self.settings_file_name = file_name

    def load_settings(self):
       
        try:
            with open(self.settings_file_name) as json_file:
                data = json.load(json_file)
        except:
            print("Error loading settings")
            return None
        return data["settings"]
