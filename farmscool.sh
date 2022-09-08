#!/bin/bash

source /home/challenge_env/virtualenv/bin/activate
python /home/challenge_env/rlberry-farms/challenge/add_xp_to_queue.py $(realpath $1) $2
