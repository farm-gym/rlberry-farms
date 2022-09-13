from redis import Redis
from rq.command import send_stop_job_command
import sys

redis = Redis()

# This will raise an exception if job is invalid or not currently executing
send_stop_job_command(redis, sys.argv[1])
