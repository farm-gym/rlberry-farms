import logging
from rq import Queue
from redis import Redis
from rlberry.experiment import load_experiment_results
from rlberry.experiment import experiment_generator
from rlberry.manager.multiple_managers import MultipleManagers
from sys import argv
from rlberry_rq.run_experiment import run_experiment
import argparse
from rlberry.utils.logging import configure_logging
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "experiment_file", type=str, help="YAML file describing the experiment"
)
parser.add_argument(
    "--queue",
    help="queue used. Possible values are low, default and high.",
    type=str,
    default="default",
)
parser.add_argument("--n-fit", help="number of fits.", type=int, default=4)
parser.add_argument(
    "--max-workers",
    help="maximum number of workers. -1 means unlimited.",
    type=int,
    default=-1,
)
parser.add_argument(
    "--output-dir", help="output directory.", type=str, default="results"
)
parser.add_argument(
    "--parallelization",
    help="type of parallelization. Possible values are process or thread.",
    type=str,
    default="process",
)
parser.add_argument("--enable-tensorboard", action="store_true")
args = parser.parse_args()

configure_logging(file_path=Path(args.output_dir) / "out.log")
logger = logging.getLogger(__name__)

redis_conn = Redis()
q = Queue(args.queue, connection=redis_conn)


experiment_kwargs = dict(
    experiment_file=args.experiment_file,
    n_fit=args.n_fit,
    max_workers=args.max_workers,
    output_dir=args.output_dir,
    parallelization=args.parallelization,
    enable_tensorboard=args.enable_tensorboard,
)

job1 = q.enqueue(run_experiment, kwargs=experiment_kwargs)
