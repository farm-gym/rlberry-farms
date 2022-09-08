import logging
from docopt import docopt
from pathlib import Path
import os
from rlberry.experiment.yaml_utils import parse_experiment_config
from rlberry.manager import AgentManager, evaluate_agents
from rlberry import check_packages
import pandas as pd
import subprocess

logger = logging.getLogger(__name__)

LEADERBOARD = "~/leaderboard.csv"


def experiment_generator(
    experiment_file,
    n_fit=4,
    max_workers=-1,
    output_dir="results",
    parallelization="process",
    enable_tensorboard=False,
):
    """
    Parse command line arguments and yields AgentManager instances.
    """
    if not os.isdir('/tmp/farm_tmp'):
        os.mkdir('/tmp/farm_tmp')
    subprocess.run(['mv', experiment_file, '/tmp/farm_tmp/latest_script.py'])
    sys.path.append( '/tmp/farm_tmp' )
    from latest_script import Agent as ContenderAgent
    
    if max_workers == -1:
        max_workers = None

    if enable_tensorboard:
        if check_packages.TENSORBOARD_INSTALLED:
            agent_manager_kwargs.update(dict(enable_tensorboard=True))
        else:
            logger.warning(
                "Option --enable_tensorboard is not available: tensorboard is not installed."
            )
    if 'eval_kwargs' in agent_manager_kwargs:
        del agent_manager_kwargs['eval_kwargs']

    yield AgentManager(**agent_manager_kwargs, eval_kwargs=dict(eval_horizon=365))


def run_experiment(
    experiment_file,
    n_fit=4,
    max_workers=-1,
    output_dir="results",
    parallelization="process",
    enable_tensorboard=False,
):
    multimanagers = MultipleManagers(
        parallelization="thread"
    )

    for agent_manager in experiment_generator(
        experiment_file,
        n_fit=8,
        max_workers=-1,
        output_dir="results_"+os.getlogin(),
        parallelization="process",
        enable_tensorboard=False,
    ):
        multimanagers.append(agent_manager)

    multimanagers.run()
    multimanagers.save()
    
    # Evaluating
    data = evaluate_agents(multimanagers.instances, n_simulations=100, show=False).values
    
    # Saving to leaderboard
    if not(os.path.isfile(LEADERBOARD)):
        df = pd.DataFrame({'name':[],'name_agent':[], 'evaluation':[]})
    else:
        df = pd.read_csv(LEADERBOARD, index_col=0)
    df = pd.concat([df, pd.DataFrame({'name':[os.getlogin()],
                                      'name_agent':[multimanagers.instances[0].agent_name],
                                      'evaluation_mean':[np.mean(data)],
                                      'evaluation_median':[np.median(data)],
                                      'evaluation_std':[np.std(data)]})])
    df = df.sort_values(by=["evaluation_mean"], ascending=False)
    df.to_csv(LEADERBOARD)
    # Deleting the manager.
    del multimanagers
