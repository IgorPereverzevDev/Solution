from datetime import datetime

from dateutil import tz
import constants


def user_opinion(args, session_count=1, day_session_count=1):
    for i, j in zip(args, args[1:]):
        diff = formatting_time(i, j)
        if diff[0] == constants.DAY:
            day_session_count += 1
        if diff[1] >= constants.MIN:
            session_count += 1
    return session_count == constants.LIMIT_SESSION and day_session_count == constants.DAY_DURATION


def formatting_time(*args):
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    # convert to time_format
    date_first = datetime.strptime(args[0], constants.TIME_FORMAT)
    date_second = datetime.strptime(args[1], constants.TIME_FORMAT)

    # replace timezone
    rep_date_first = date_first.replace(tzinfo=from_zone)
    rep_date_second = date_second.replace(tzinfo=from_zone)

    timezone_date_first = rep_date_first.astimezone(to_zone)
    timezone_date_second = rep_date_second.astimezone(to_zone)

    # calculate the difference
    diff = abs(timezone_date_first - timezone_date_second)
    diff_min = diff.days * 1440 + diff.seconds / 60
    diff_day = date_second.day - date_first.day

    return diff_day, diff_min
