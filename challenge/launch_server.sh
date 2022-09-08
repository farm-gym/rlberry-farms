#!/bin/bash
source /home/challenge_env/virtualenv/bin/activate

systemctl start redis.service
python3 rq_worker.py
systemctl  stop redis.service
