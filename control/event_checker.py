from __future__ import print_function
import json
import datetime
import time
import pickle
import os.path
import re
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from time_handler import TimeHandler

# If modifying these scopes, delete the file token.pickle.


class EventChecker:

    def __init__(self):

     
        SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

        sleeping_time = 1800
    def load_events(self):
        """Shows basic usage of the Google Calendar API.
        Prints the start and name of the next 10 events on the user's calendar.
        """
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        try:
            service = build('calendar', 'v3', credentials=creds)
        except:
            print("couldnt build service")
            return
        # Call the Calendar API
        now = datetime.datetime.utcnow().isoformat() + 'Z' # 'Z' indicates UTC time
        try:
            events_result = service.events().list(calendarId='primary', timeMin=now,
                                                maxResults=1, singleEvents=True,
                                                orderBy='startTime').execute()
            events = events_result.get('items', [])
            return events
        except:
            print("couldnt't get events")
            return None
    def next_heat_up_event(self):
        events = self.load_events()
        return_dict = {"hours_to_event": None, "degree_target": None}

        if events:
            for e in events:
                if re.match('^.*boiler heat up at (\d+) degrees$', e['summary']):
                    degree_target = int(re.split('^.*boiler heat up at (\d+) degrees$', e['summary'])[1])
                    start = self.date_to_datetime(e['start'].get('dateTime', e['start'].get('date')))
                    time_to_event  = (start - (datetime.datetime.now() + datetime.timedelta(hours= 1) )) / datetime.timedelta(hours=1)


                    if (time_to_event > 0):
                        
                        return_dict['hours_to_event'] = time_to_event 
                        return_dict['degree_target'] = degree_target
                    break
        return return_dict

    def check_off_event(self):
        events = self.load_events()
        if not events:
            return False
        else:
            for e in events:

                if("#off" in e['summary']):
                    start = self.date_to_datetime(e['start'].get('dateTime', e['start'].get('date')))
                    end = self.date_to_datetime( e['end'].get('dateTime', e['end'].get('date')))
                    
                    return self.is_date_between(start, end)
            return False

    def date_to_datetime(self, date):
        return datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S+01:00")

    def is_date_between(self, begin_date, end_date):
        check_date = datetime.datetime.now()   

        return check_date >= begin_date or check_date <= end_date - datetime.timedelta(hours=3)

if __name__ == '__main__':
    e = EventChecker()
    e.check_event()

    

