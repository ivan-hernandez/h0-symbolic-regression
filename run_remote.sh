#!/bin/bash
cd ~/hubble_pilot
exec python3 -u hubble_pilot.py --real-only > output/run3.log 2>&1
