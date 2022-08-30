#!/bin/bash

systemctl start redis.service
rq worker high default low
systemctl  stop redis.service
