#!/bin/bash
source /home/challenge_env/virtualenv/bin/activate

systemctl start redis.service
python rq_worker.py
systemctl  stop redis.service
