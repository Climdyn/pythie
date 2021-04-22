
import numpy as np
import datetime


def convert_time_to_int(time):
    if isinstance(time, datetime.timedelta):
        return time.days * 24 + time.seconds / 3600
    elif isinstance(time, datetime.datetime):
        return time.hour
    else:
        return None


def map_times_to_int_array(times):
    return np.array((list(map(convert_time_to_int, times))))
