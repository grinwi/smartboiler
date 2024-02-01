##########################################################
# Bachelor's thesis                                      #
# From a dumb boiler to a smart one using a smart socket #
# Author: Adam Grünwald                                  #
# BUT FIT BRNO, Faculty of Information Technology        #
# 26/6/2021                                              #
#                                                        #
# Module with class used for comunicating with Google    #
# Calendar API and searching for special events.         #
##########################################################

from __future__ import print_function
import json
import datetime
import time
import pickle
import os.path
import re
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request

from time_handler import TimeHandler


class EventChecker:

    def __init__(self):

        self.SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
        self.TimeHandler = TimeHandler()

    def load_events(self):
        """Loads events from Google Calendar API using credentials of Google Calendar.

        Code for communicating with Google Calendar API is inspired by a manual on https://developers.google.com/calendar/
        quickstart/python

        Returns:
            [list]: [list of events]
        """

        creds = None

        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        try:
            service = build('calendar', 'v3', credentials=creds)
        except:
            print("couldn't build service")
            return
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        try:
            events_result = service.events().list(calendarId='primary', timeMin=now,
                                                  maxResults=1, singleEvents=True,
                                                  orderBy='startTime').execute()
            events = events_result.get('items', [])
            return events
        except:
            print("couldn't get events")
            return None

    def next_calendar_heat_up_event(self, Boiler):
        """Search next event in a calendar which contains specific words describing the process of heating up.

        Args:
            Boiler ([class]): [class of boiler]

        Returns:
            [dict]: [dictionary describing next heating up event]
        """
        events = self.load_events()
        return_dict = {"hours_to_event": None, "degree_target": None}

        if events:
            for e in events:
                if re.match('^.*boiler heat up at (\d+) degrees$', e['summary']):
                    degree_target = int(
                        re.split('^.*boiler heat up at (\d+) degrees$', e['summary'])[1])
                    start = self.TimeHandler.date_to_datetime(
                        e['start'].get('dateTime', e['start'].get('date')))
                    time_to_event = (start - (datetime.datetime.now() +
                                     datetime.timedelta(hours=1))) / datetime.timedelta(hours=1)

                    if (time_to_event > 0):

                        return_dict['hours_to_event'] = time_to_event
                        return_dict['degree_target'] = degree_target
                    break
                if re.match('^.*Prepare (\d+) showers$', e['summary']):
                    number_of_showers = int(
                        re.split('^.*Prepare (\d+) showers$', e['summary'])[1])

                    degree_target = Boiler.showers_degrees(
                        number_of_showers=number_of_showers)

                    start = self.TimeHandler.date_to_datetime(
                        e['start'].get('dateTime', e['start'].get('date')))
                    time_to_event = (start - (datetime.datetime.now() +
                                     datetime.timedelta(hours=1))) / datetime.timedelta(hours=1)

                    if (time_to_event > 0):

                        return_dict['hours_to_event'] = time_to_event
                        return_dict['degree_target'] = degree_target
                    break

        return return_dict

    def check_off_event(self):
        """Search for an event for turning off the boiler.

        Returns:
            [type]: [description]
        """
        events = self.load_events()
        if not events:
            return False
        else:
            for e in events:

                if("#off" in e['summary']):
                    start = self.TimeHandler.date_to_datetime(
                        e['start'].get('dateTime', e['start'].get('date')))
                    end = self.TimeHandler.date_to_datetime(
                        e['end'].get('dateTime', e['end'].get('date')))

                    return self.TimeHandler.is_date_between(start, end)
            return False


if __name__ == '__main__':
    e = EventChecker()
    e.check_event()
