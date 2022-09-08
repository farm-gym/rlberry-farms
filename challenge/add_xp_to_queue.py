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

parser = argparse.ArgumentParser()

parser.add_argument(
    "experiment_file", type=str, help="Python file with "
)
parser.add_argument(
    "--queue",
    help="queue used. Possible values are low, default and high.",
    type=str,
    default="default",
)

parser.add_argument("--enable-tensorboard", action="store_true")
args = parser.parse_args()

output_dir = "results"

configure_logging(file_path=Path(output_dir) / "out.log")
logger = logging.getLogger(__name__)

redis_conn = Redis()
q = Queue(args.queue, connection=redis_conn)


experiment_kwargs = dict(
    experiment_file=args.experiment_file,
    n_fit=1,
    output_dir=output_dir,
    parallelization="process",
    enable_tensorboard=args.enable_tensorboard,
)

job1 = q.enqueue(run_experiment, kwargs=experiment_kwargs)
