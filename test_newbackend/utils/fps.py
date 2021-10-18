"""simple class to calc frames per second
https://stackoverflow.com/a/54539292/8615419
"""

import collections
import time

class FPS:
    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)
    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return round(len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0]), 2)
        else:
            return 0.0