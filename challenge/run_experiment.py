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

import datetime

logger = rlberry.logger

LEADERBOARD = "leaderboard.csv"
ARCHIVE_DIR = "/media/data1/challenge"
LOGFILE_LOC = "/home/challenge_env/rlberry-farms/challenge/"

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
    if not os.path.isdir("/tmp/farm_tmp"):
        os.mkdir("/tmp/farm_tmp")
    subprocess.run(["cp", agent_file, "/tmp/farm_tmp/latest_script.py"])
    sys.path.append("/tmp/farm_tmp")
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
    if not (os.path.isfile(LEADERBOARD)):
        df = pd.DataFrame()
    else:
        df = pd.read_csv(LEADERBOARD, index_col=0)
    new_score = pd.DataFrame(
        {
            "name": [name],
            "name_agent": [agent_manager.agent_name],
            "evaluation_mean": [np.mean(data)],
            "evaluation_median": [np.median(data)],
            "evaluation_std": [np.std(data)],
            "time (m)":[time_to_completion/60]
        }
    )

    # keep only max for leaderboard
    df = pd.concat([df, new_score], ignore_index=True)
    df2 = pd.DataFrame()
    for name in df['name'].unique():
        dfname = df.loc[df['name']==name]
        df2 =df2.append(dfname.iloc[np.argmax(dfname['evaluation_mean'])], ignore_index=True)
    
    df = df2.reset_index()
    
    df = df2.sort_values(by=["evaluation_mean"], ascending=False)
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

    date_now = now.strftime("%d/%m/%Y %H:%M:%S")

    subprocess.run(["cp", os.path.join(LOGFILE_LOC,"logfile.log"), os.path.join(output_dir, date_now+"_logfile.log")])
    with open(os.path.join(LOGFILE_LOC,"logfile.log"), 'w') as f:
        f.write('')
    
    # Deleting the manager.
    del agent_manager
