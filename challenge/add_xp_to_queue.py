import logging
from rq import Queue
from redis import Redis
from rlberry.experiment import load_experiment_results
from rlberry.experiment import experiment_generator
from rlberry.manager.multiple_managers import MultipleManagers
from sys import argv
from run_experiment import run_experiment
import argparse
from rlberry.utils.logging import configure_logging
from pathlib import Path
import os

parser = argparse.ArgumentParser()

parser.add_argument("agent_file", type=str, help="Python file with Agent class in it ")

parser.add_argument("budget", type=int, help="Number of steps", default=1000)

parser.add_argument("name", type=str, help="Name of submition", default="Anon")

parser.add_argument(
    "--farm",
    help="farm used.",
    type=int,
    default=0,
)

parser.add_argument("--enable-tensorboard", action="store_true")
args = parser.parse_args()

configure_logging(file_path=Path(output_dir) / "out.log")
logger = logging.getLogger(__name__)

redis_conn = Redis()
q = Queue("default", connection=redis_conn)


experiment_kwargs = dict(
    agent_file=args.agent_file,
    budget=args.budget,
    n_fit=1,
    parallelization="process",
    enable_tensorboard=args.enable_tensorboard,
    farm=args.farm,
    name=args.name,
)

job1 = q.enqueue(
    run_experiment, kwargs=experiment_kwargs, job_timeout=3 * 3600
)  # limit job to 3h
