import logging
from rq import Queue
from redis import Redis
from rlberry.experiment import load_experiment_results
from rlberry.experiment import experiment_generator
from rlberry.manager.multiple_managers import MultipleManagers
from sys import argv
from run_experiment import run_experiment
import argparse
from pathlib import Path
import os

CHALLENGE_DIR = "/challenge_bin"
DATA_LOC = os.environ.get('challenge_data_dir')

parser = argparse.ArgumentParser()

parser.add_argument("agent_file", type=str, help="Python file with Agent class in it ")

parser.add_argument("budget", type=int, help="Number of steps", default=1000)

parser.add_argument("name", type=str, help="Name of submition", default="Anon")

parser.add_argument(
    "--farm",
    help="farm used.",
    type=int,
    default=1,
)

parser.add_argument("--enable-tensorboard", action="store_true")
args = parser.parse_args()

redis_conn = Redis()
q = Queue("default", connection=redis_conn)


experiment_kwargs = dict(
    agent_file=args.agent_file,
    budget=args.budget,
    n_fit=15,
    parallelization="process",
    enable_tensorboard=args.enable_tensorboard,
    farm=args.farm,
    name=args.name,
)
with open(os.path.join(DATA_LOC,"logfile.log"), 'w') as f:
    f.write('')
job = q.enqueue(
    run_experiment, kwargs=experiment_kwargs, job_timeout=3 * 3600
)  # limit job to 3h

