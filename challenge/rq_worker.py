#!/usr/bin/env python
import sys
from rq import Connection, Worker
import logging
import rlberry



if __name__ == "__main__":

    logger = logging.getLogger("rq.worker")

    with Connection():
        logger.info("Launching queue server")
        qs = ["default"]
        w = Worker(qs)
        w.work()
