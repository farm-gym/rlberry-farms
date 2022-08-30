# rlberry farms

This repository aim at using games constructed from [Farm-Gym](https://gitlab.inria.fr/rl4ae/farm-gym/) using [rlberry](https://github.com/rlberry-py/rlberry).

A starter code using PPO is available in `examples/ppo_farm0.py`.

## Installation

To install this repo, one can use pip. From the root of the repo

```bash
pip install -e .
pip install git+https://github.com/rlberry-py/rlberry
git clone https://gitlab.inria.fr/rl4ae/farm-gym ~/farm-gym
pip install  -e ~/farm-gym
```

WARNING : the `-e` in the above commands are mandatory as the YAML files are missing otherwise.
