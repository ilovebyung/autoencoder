# pip install python-dateutil
from dateutil import parser
start = "Tue, 16 Nov 2021 00:00:00 GMT"
end = "Tue, 16 Nov 2021 00:00:00 GMT"

dt = parser.parse(start)
str(dt.date)
