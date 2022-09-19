#!/bin/bash

sudo su challenger 
singularity instance start --bind data:/mnt challenge.sif challenge

