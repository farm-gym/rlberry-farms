#!/bin/bash

sudo su challenger -c "singularity instance start  --bind /home/challenger/data:/mnt /home/challenger/challenge.sif challenge"
