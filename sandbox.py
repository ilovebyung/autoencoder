# pip install python-dateutil
from dateutil import parser
import datetime

start = "Tue, 16 Nov 2021 00:00:00 GMT"
end = "Tue, 16 Nov 2021 00:00:00 GMT"


start = parser.parse(start)
end = parser.parse(end)

DT = datetime.datetime(2003, 8, 1, 12, 4, 5)
DT += datetime.timedelta(days=1)
print(DT)


start - end
adjusted_day = start.day + 1


class DatePicker(start, end):
    if (start == end):
