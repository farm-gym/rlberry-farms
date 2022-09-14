#!/usr/bin/env python
import sys
from rq import Connection, Worker
import logging
import rlberry


logger = logging.getLogger("rq.worker")
logger2 = rlberry.logger

class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != "\n":
            self.level(message)

    def flush(self):
        pass


sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.info)

fh = logging.FileHandler("logfile.log")
formatter = logging.Formatter("")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger2.addHandler(fh)


if __name__ == "__main__":
    with Connection():
        logger.info("Launching queue server")
        qs = ["default"]

        w = Worker(qs)
        w.work()
