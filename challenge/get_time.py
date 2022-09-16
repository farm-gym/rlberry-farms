import numpy as np
import re
import os
import datetime
from dateutil.parser import parse
from datetime import timedelta

today = str(datetime.datetime.today().date())
LOG_FILE = "logfile.log"

def get_eta():
    if os.path.isfile(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            text = f.read()
        total_budget = int(re.search(r'(?<= budget\=)\w+', text).group(0))
        steps = re.findall(r'(?<=global_step = )\w+', text)
        current_step = int(steps[-1])

        begin_date = text[:8]
        begin_datetime = parse(today + ' '+ begin_date)

        now = datetime.datetime.now()
        time_passed = int((now-begin_datetime).total_seconds())
        mean_sec_per_steps = time_passed/current_step

        eta = mean_sec_per_steps*(total_budget-current_step)
        return (eta //60, eta - eta // 60)
    else:
        return None
