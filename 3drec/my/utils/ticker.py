from datetime import date, time, datetime, timedelta
from time import sleep


class IntervalTicker():
    def __init__(self, interval=60):
        self.interval = timedelta(seconds=interval)
        self.last_tick = datetime.now()
        self.now = self.last_tick

    def tick(self):
        self.now = datetime.now()
        if (self.now - self.last_tick) > self.interval:
            self.last_tick = self.now
            return True

    def tick_str(self):
        return self.now.isoformat(timespec='seconds')
