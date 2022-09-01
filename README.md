# rlberry farms

This repository aim at using games constructed from [Farm-Gym](https://gitlab.inria.fr/rl4ae/farm-gym/) using [rlberry](https://github.com/rlberry-py/rlberry).

A starter code using PPO is available in `examples/ppo_farm0.py`.

# Challenge
 
This package is intended to be used for the internal SCOOL-team challenge on farmgym. Included in this repo are the script for job submission to the challenge. 
  
## Installation

To install this repo, one can use pip. It is advised to use a virtual environment in order to avoid conflicting library (see [python website](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)).

Then, 

```bash
pip install git+https://gitlab.inria.fr/scool/rlberry-farms
```

## Job submission

TODO


# For challenge maintainer
In order to launch the server for the challenge, `redis` must be installed on the server. See basic usage of queue in https://github.com/TimotheeMathieu/rlberry-queue

