import os
import argparse
from galaxyGeniusMSA.config import Configuration

parser = argparse.ArgumentParser()

parser.add_argument('-w', '--workspace', type=str, default='workspace')

args = parser.parse_args()

workspace = args.workspace

os.makedirs(workspace, exist_ok=True)

configuration = Configuration()
configuration.init(workspace=workspace)

print(f'Configuration files are created in {workspace}. Please edit them!')
