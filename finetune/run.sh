#!/usr/bin/bash

accelerate launch --config_file=deepspeed-zero2.yaml tune.py