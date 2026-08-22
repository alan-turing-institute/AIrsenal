"""Gameweek facts that only the FPL API knows."""

from datetime import datetime

from airsenal.fetch.fpl_api import get_fetcher


def get_last_finished_gameweek() -> int:
    """
    Query the API to see what the last gameweek marked as 'finished' is.
    """
    event_data = get_fetcher().get_event_data()
    last_finished = 0
    for gw in sorted(event_data.keys()):
        if event_data[gw]["is_finished"]:
            last_finished = gw
        else:
            return last_finished
    return last_finished


def is_transfer_deadline_today() -> bool:
    """
    Return True if there is a transfer deadline later today.
    """
    deadlines = get_fetcher().get_transfer_deadlines()
    for deadline in deadlines:
        deadline_datetime = datetime.strptime(deadline, "%Y-%m-%dT%H:%M:%SZ")
        if (deadline_datetime - datetime.now()).days == 0:
            return True
    return False
