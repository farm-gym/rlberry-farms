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


logger = rlberry.logger

LEADERBOARD = "leaderboard.csv"
ARCHIVE_DIR = "/media/data1/challenge"


def get_farm(farm):
    if farm == 0:
        return Farm0, {}
    elif farm == 1:
        return Farm1, {}
    else:
        raise RuntimeError("No such farm")


def experiment_generator(
    agent_file=None,
    n_fit=4,
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
        eval_kwargs=dict(eval_horizon=365),
        **kwargsii
    )


def run_experiment(
    agent_file,
    budget=100,
    n_fit=4,
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
        max_workers=-1,
        parallelization="process",
        enable_tensorboard=enable_tensorboard,
        output_dir=output_dir,
    )
    agent_manager.fit()

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
        }
    )

    # keep only max for leaderboard
    if name in df["name"]:
        if df.loc[df["name"] == name, "evaluation_mean"] < new_score["evaluation_mean"]:
            for key in ["evaluation_mean", "evaluation_median", "evaluation_std"]:
                df.loc[df["name"] == name, key] = new_score[key]
    else:
        df = pd.concat([df, new_score], ignore_index=True)

    df = df.sort_values(by=["evaluation_mean"], ascending=False)
    df.to_csv(LEADERBOARD)

    # Archiving data
    archive_scores = os.path.join(ARCHIVE_DIR, "all_scores.csv")
    if not (os.path.isfile(archive_scores)):
        df = pd.DataFrame()
    else:
        df = pd.read_csv(archive_scores, index_col=0)
    df = pd.concat([df, new_score], ignore_index=True)
    df.to_csv(archive_scores)

    # Deleting the manager.
    del agent_manager
