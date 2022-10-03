import logging
from docopt import docopt
from pathlib import Path
import os
import sys
from rlberry.experiment.yaml_utils import parse_experiment_config
from rlberry.manager import AgentManager, evaluate_agents
from rlberry import check_packages
import pandas as pd
import subprocess

import rlberry
from rlberry_farms import Farm0, Farm1
import numpy as np
import time
import importlib

import datetime

logger = rlberry.logger

CHALLENGE_DIR = "/mnt/data/challenge"
DATA_DIR = "/mnt/data/challenge"

LEADERBOARD = os.path.join(DATA_DIR,"leaderboard.csv")
ARCHIVE_DIR = DATA_DIR
LOGFILE_LOC = DATA_DIR

def get_farm(farm):
    if farm == 0:
        return Farm0, {}
    elif farm == 1:
        return Farm1, {}
    else:
        raise RuntimeError("No such farm")

def experiment_generator(
    agent_file=None,
    n_fit=15,
    budget=100,
    farm=None,
    max_workers=-1,
    output_dir="results",
    parallelization="process",
    enable_tensorboard=False,
):
    sys.path.append(DATA_DIR)
    try:
        from latest_script import Agent as ContenderAgent
    except:
        raise RuntimeError("Import of Agent failed")
    if max_workers == -1:
        max_workers = None
    kwargsii = {}
    if enable_tensorboard:
        if check_packages.TENSORBOARD_INSTALLED:
            kwargsii.update(dict(enable_tensorboard=True))
        else:
            logger.warning(
                "Option --enable_tensorboard is not available: tensorboard is not installed."
            )

    return AgentManager(
        ContenderAgent,
        farm,
        fit_budget=budget,
        output_dir=output_dir,
        n_fit = n_fit,
        init_kwargs={"compress_pickle":True},
        parallelization="process",
        mp_context="spawn",
        eval_kwargs=dict(eval_horizon=365),
        default_writer_kwargs={"style_log":"one_line"},
        **kwargsii
    )


def run_experiment(
    agent_file,
    budget=100,
    n_fit=15,
    max_workers=-1,
    parallelization="process",
    enable_tensorboard=False,
    farm=0,
    name="Anon",
):
    output_dir = os.path.join(ARCHIVE_DIR, name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    agent_manager = experiment_generator(
        agent_file=agent_file,
        budget=budget,
        farm=get_farm(farm),
        n_fit=n_fit,
        parallelization="process",
        enable_tensorboard=enable_tensorboard,
        output_dir=output_dir,
    )
    a = time.time()
    agent_manager.fit()
    time_to_completion = time.time()-a
    # Evaluating
    data = evaluate_agents([agent_manager], n_simulations=100, show=False).values

    # Saving to leaderboard
    now = datetime.datetime.now()

    date_now = now.strftime("%d_%m_%Y %H:%M:%S")
    if not (os.path.isfile(LEADERBOARD)):
        df = pd.DataFrame()
    else:
        df = pd.read_csv(LEADERBOARD, index_col=0)
    new_score = pd.DataFrame(
        {
            "name": [name],
            "name_agent": [agent_manager.agent_name],
            "eval_mean": [np.mean(data)],
            "eval_median": [np.median(data)],
            "eval_std": [np.std(data)],
            "time (m)":[time_to_completion/60],
            "date":[date_now]
        }
    )

    # keep only max for leaderboard
    df = pd.concat([df, new_score], ignore_index=True)
    df2 = pd.DataFrame()
    for name in df['name'].unique():
        dfname = df.loc[df['name']==name]
        df2 =df2.append(dfname.iloc[np.argmax(dfname['eval_mean'])], ignore_index=True)

    df = df2.reset_index()

    df = df2.sort_values(by=["eval_mean"], ascending=False)
    df.to_csv(LEADERBOARD)

    # Archiving data
    archive_scores = os.path.join(ARCHIVE_DIR, "all_scores.csv")
    if not (os.path.isfile(archive_scores)):
        df = pd.DataFrame()
    else:
        df = pd.read_csv(archive_scores, index_col=0)
    df = pd.concat([df, new_score], ignore_index=True)
    df.to_csv(archive_scores)

    # copy logs

    now = datetime.datetime.now()

    date_now = now.strftime("%d_%m_%Y-%H:%M:%S")

    subprocess.run(["cp", os.path.join(LOGFILE_LOC,"logfile.log"), os.path.join(output_dir, date_now+"_logfile.log")])
    with open(os.path.join(LOGFILE_LOC,"logfile.log"), 'w') as f:
        f.write('')

    # Deleting the manager.
    del agent_manager
