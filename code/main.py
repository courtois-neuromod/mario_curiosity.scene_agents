import os
import sys
import retro
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from scipy.special import softmax
import pandas as pd
import argparse

import mario_scenes

sys.path.append(os.path.join(os.getcwd()))



from load_data import get_mastersheet, get_models, parse_state_files
from utils import process_state, filter_states


from src.ppo.env import preprocess_frames, complex_movement_to_button_presses
from src.ppo.emulation import add_unused_buttons
from src.ppo import PPO

def main(args):

    models_ppo = get_models(args.models)
    states = parse_state_files(args.datapath)

    filters = {
        'sub': args.sub,
        'ses': args.ses,
        'level': args.level,
        'scene': args.scene
    }

    filtered_states = filter_states(states, filters)
    info_scenes = get_mastersheet(args.mastersheet)

    for i, ppo_row in models_ppo.iterrows():
        for y , state_row in filtered_states.iterrows():
            process_state(state_row, ppo_row, info_scenes)
            break
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract .state files from Mario dataset and information about the scenes")
    parser.add_argument(
        "-d",
        "--datapath",
        default='sourcedata/',
        type=str,
        help="Data path to look for the mastersheet and .state files. Should be the root of the Mario dataset.",
    )
    parser.add_argument(
        "-m",
        "--models",
        default= os.path.join('sourcedata', 'models'),
        type=str,
        help="Path to the models folder, where the PPO models are stored.",
    )
    parser.add_argument(
        "-ms",
        "--mastersheet",
        default= os.path.join('sourcedata', 'scenes_mastersheet.csv'),
        type=str,
        help="Path to the models folder, where the PPO models are stored.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default='outputdata/',
        type=str,
        help="Path to the derivatives folder, where the outputs will be saved.",
    )
    parser.add_argument(
        "-sp",
        "--stimuli",
        default=None,
        type=str,
        help="Path to the stimuli folder containing the game ROMs. Defaults to <datapath>/stimuli if not specified.",
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        default=1,
        type=int,
        help="Number of CPU cores to use for parallel processing.",
    )
    parser.add_argument(
        '--subjects', 
        '-sub', 
        nargs='+',
        default=None,
        help='List of subjects to process (e.g., sub-01 sub-02). If not specified, all subjects are processed.'
    )
    parser.add_argument(
        '--sessions', 
        '-ses', 
        nargs='+', 
        default=None,
        help='List of sessions to process (e.g., ses-001 ses-002). If not specified, all sessions are processed.'
    )
    parser.add_argument(
        '--levels', 
        '-l', 
        nargs='+', 
        default=None,
        help='List of level to process (e.g., w1l1 w1l2). If not specified, all sessions are processed.'
    )
    parser.add_argument(
        '--sessions', 
        '-ses', 
        nargs='+', 
        default=None,
        help='List of sessions to process (e.g., scene-1 scene-2). If not specified, all sessions are processed.'
    )

    args = parser.parse_args()
    

args = parser.parse_args()
main(args)





""""
def process_state(model, state, INFO_SCENES):
    blabla process that stuff

def main(args):
    subjects = args.subjects
    models = args.models
    levels = args.levels
    sourcedata_path = args.sourcedata_path
    outputdata_path = args.outputdata_path

    INFO_SCENES = get_mastersheet(path='/home/hugo/simexp/mario.scenes/resources/scenes_mastersheet.csv')

    bk2_files = list(root_folder.rglob("*.bk2"))
    
    for subj in subjects:
        for level in levels:
            for model in models:
                # Filter the list of bk2
                bk2_files_filtered = [
                    bk2_file for bk2_file in bk2_files
                    if f"{subj}" in str(bk2_file) and f"{level}" in str(bk2_file)
                ]
                for bk2_file in bk2_files_filtered:
                    # Process each file in the selected bk2s
                    process_state(model, state, INFO_SCENES)


"""
