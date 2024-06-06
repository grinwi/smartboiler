# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# This module is used for communicating with Google Calendar API and searching for special events.


from __future__ import print_function
from datetime import datetime, timezone, timedelta
import pickle
import os.path
import re

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from time_handler import TimeHandler


class EventChecker:
    """Class for communicating with Google Calendar API and searching for special events."""

    def __init__(self):
        """Initialize the class with the scopes for Google Calendar API and TimeHandler class."""
        self.SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
        self.TimeHandler = TimeHandler()

    def load_events(self) -> list:
        """Loads events from Google Calendar API using credentials of Google Calendar.

        Code for communicating with Google Calendar API is inspired by a manual on https://developers.google.com/calendar/
        quickstart/python

        Returns:
            [list]: [list of events]
        """
        try:

            creds = None
            token_file = "/app/token.json"
            token_file = "token.json"
            creds_file = "/app/credentials.json"

            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
            # If there are no (valid) credentials available, let the user log in.
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        creds_file, self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open(token_file, "w") as token:
                    token.write(creds.to_json())

            service = build("calendar", "v3", credentials=creds)

            now = datetime.now().isoformat() + "Z"

            events_result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    maxResults=1,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = events_result.get("items", [])
            return events
        except Exception as e:
            print("Error while loading events")
            print(e)
            return None

    def next_calendar_heat_up_event(self) -> dict:
        """Search next event in a calendar which contains specific words describing the process of heating up.

        Args:

        Returns:
            [dict]: [dictionary describing next heating up event]
        """
        events = self.load_events()
        return_dict = {"minutes_to_event": None, "degree_target": None}

        if events:
            for e in events:
                if re.match("^.*heat water at (\d+) degrees$", e["summary"]):
                    degree_target = int(
                        re.split("^.*heat water at (\d+) degrees$", e["summary"])[1]
                    )
                    start = self.TimeHandler.date_to_datetime(
                        e["start"].get("dateTime", e["start"].get("date"))
                    )
                    end = self.TimeHandler.date_to_datetime(
                        e["end"].get("dateTime", e["end"].get("date"))
                    )
                    time_to_event = (
                        start - datetime.now(timezone.utc)
                    ) / timedelta(hours=1)
                    time_to_end_event = (
                        end - datetime.now(timezone.utc)
                    ) / timedelta(hours=1)

                    if time_to_event > 0:

                        return_dict["minutes_to_event"] = time_to_event * 60
                        return_dict["degree_target"] = degree_target

                        return return_dict
                    # case in event
                    elif time_to_end_event > 0:
                        return_dict["minutes_to_event"] = 0
                        return_dict["degree_target"] = degree_target

                        return return_dict

        return return_dict

    def check_off_event(self) -> bool:
        """Search for an event for turning off the boiler.

        Returns:
            [bool]: [if the turn offevent is happening now or not]
        """
        events = self.load_events()
        if events:
            for e in events:

                if "#off" in e["summary"]:
                    start = self.TimeHandler.date_to_datetime(
                        e["start"].get("dateTime", e["start"].get("date"))
                    )
                    end = self.TimeHandler.date_to_datetime(
                        e["end"].get("dateTime", e["end"].get("date"))
                    )
                    return self.TimeHandler.is_date_between(start, end)
        return False


if __name__ == "__main__":
    e = EventChecker()
    print(e.check_off_event())
