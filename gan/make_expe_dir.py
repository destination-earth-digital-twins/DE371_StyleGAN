#!/usr/bin/env python3

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type = str, default='exp_0')

config = parser.parse_args()
cwd = os.getcwd()
if not os.path.exists(config.name):
    os.mkdir(config.name)
    os.chdir(config.name)
    os.mkdir('log')
    os.mkdir('models')
    os.mkdir('samples')
os.chdir(cwd)
