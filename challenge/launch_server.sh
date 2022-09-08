#!/bin/bash

systemctl start redis.service
python rq_worker.py
systemctl  stop redis.service
