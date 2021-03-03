from __future__ import print_function
import json
import datetime
import time
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from time_handler import TimeHandler


# If modifying these scopes, delete the file token.pickle.


class EventChecker:

    def __init__(self):
        print("------------------------------------------------------\n")
        print('initializing of EventChecker\n')
        print("------------------------------------------------------\n")


        self.SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
        self.TimeHandler = TimeHandler()

    def check_event(self):
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
            return False
        # Call the Calendar API
        now = datetime.datetime.utcnow().isoformat() + 'Z' # 'Z' indicates UTC time
        events_result = service.events().list(calendarId='primary', timeMin=now,
                                            maxResults=1, singleEvents=True,
                                            orderBy='startTime').execute()
        event = events_result.get('items', [])

        return_value = False
        if not event:
            return return_value
        else:
            for e in event:

                if("#off" in e['summary']):
                    start = self.TimeHandler.date_to_datetime(e['start'].get('dateTime', e['start'].get('date')))
                    end = self.TimeHandler.date_to_datetime( e['end'].get('dateTime', e['end'].get('date')))
                    
                    if(self.TimeHandler.is_date_between(start, end)):
                        return_value = True
            return return_value


if __name__ == '__main__':
    e = EventChecker()
    e.check_event()

    

