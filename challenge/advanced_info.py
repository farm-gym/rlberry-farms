from redis import Redis
from rq.command import send_stop_job_command
import sys
from rq.job import Job
from rq import Queue
from rq.registry import StartedJobRegistry, ScheduledJobRegistry
import pandas as pd
from tabulate import tabulate
import re

import numpy as np
import re
import os
import datetime
from dateutil.parser import parse
from datetime import timedelta

today = str(datetime.datetime.today().date())
LOG_FILE = "/mnt/data/challenge/logfile.log"

def get_eta():
    if os.path.isfile(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            text = f.read()
        total_budget = int(re.search(r'(?<=budget=)\w+', text).group(0))
        steps = re.findall(r'(?<=global_step = )\w+', text)
        current_step = np.max(np.array(steps[-1]).astype(int))

        begin_date = text[:8]
        begin_datetime = parse(today + ' '+ begin_date)
        now = datetime.datetime.now()
        time_passed = int((now-begin_datetime).total_seconds())
        mean_sec_per_steps = time_passed/current_step

        eta = mean_sec_per_steps*(total_budget-current_step)
        return eta //60, int(eta - (eta // 60)*60)
    else:
        return None



redis = Redis()
registry = StartedJobRegistry(name = "default", connection=redis)

df = pd.DataFrame()

for job_id in  registry.get_job_ids():
    job = Job.fetch(job_id, connection=redis)
    desc = job.description
    name = re.search(r"(?<=name=')\w+", desc).group(0)
    try:
        eta = get_eta()
        eta_fmt = '{} min and {} sec'.format(eta[0], eta[1])
    except:
        eta_fmt = None
    df = pd.concat([df, pd.DataFrame({'submitter' : [name],
                                      "id":job_id,
                                      'heartbeat' : [job.last_heartbeat],
                                      'status' : [job.get_status(refresh=True)],
                                      'Training eta (approx)': [eta_fmt]}
                                     )], ignore_index = True)



queue = Queue('default', connection=redis)
for job_id in queue.job_ids:
    job = Job.fetch(job_id, connection=redis)
    desc = job.description
    name = re.search(r"(?<=name=')\w+", desc).group(0)
    df = pd.concat([df, pd.DataFrame({'submitter' : [name],
                                      "id":job_id,
                                      'heartbeat' : [job.last_heartbeat],
                                      'status' : [job.get_status(refresh=True)]})], ignore_index = True)

    
df = df.reset_index()

print(tabulate(df, headers="keys", tablefmt="psql"))

print('Which job do you want to kill ? (enter its index)')
idx = input()

if (idx is not None) and (len(idx)>0):
    job_id = df.iloc[int(idx)]['id']
    try:
        send_stop_job_command(redis, job_id)
    except:
        job = Job.fetch(job_id, connection=redis)
        job.cancel()
        job.delete()

