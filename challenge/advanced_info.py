from redis import Redis
from rq.command import send_stop_job_command
import sys
from rq.job import Job
from rq import Queue
from rq.registry import StartedJobRegistry, ScheduledJobRegistry
import pandas as pd
from tabulate import tabulate
import re

redis = Redis()
registry = StartedJobRegistry(name = "default", connection=redis)


df = pd.DataFrame()

for job_id in  registry.get_job_ids():
    job = Job.fetch(job_id, connection=redis)
    desc = job.description
    name = re.search(r"(?<=name=')\w+", desc).group(0)
    df = pd.concat([df, pd.DataFrame({'submitter' : [name],
                                      "id":job_id,
                                      'heartbeat' : [job.last_heartbeat],
                                      'status' : [job.get_status(refresh=True)]})], ignore_index = True)



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

