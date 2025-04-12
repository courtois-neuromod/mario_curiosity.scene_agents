import os
import sys
import h5py
import numpy as np
import argparse
from pathlib import Path
<<<<<<< HEAD
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib  # Import from the official tqdm-joblib package
=======
from datetime import datetime
>>>>>>> refs/remotes/origin/main


sys.path.append(os.path.join(os.getcwd()))


from load_data import get_mastersheet, get_models, parse_state_files, get_scene, get_xpos_max
from utils import process_state, filter_states


def process_single_state(row_state, ppo_row, info_scenes, args):
    """Process a single state file with the given model."""
    state = row_state['state_path']
    sub = row_state['sub']
    ses = row_state['ses']
    model = ppo_row['loaded_models']
    scene = get_scene(state)
    x_max = get_xpos_max(info_scenes, scene)
    path_output = os.path.join(args.output, ppo_row['name_models'].split('.')[0], sub, ses, 'beh')
    
    return process_state(state, args, model, x_max, path_output, args.stimuli, verbose=args.verbose)

<<<<<<< HEAD

def main(args):

    now = datetime.now()

    # Conversion en string
    date_str = now.strftime("%Y-%m-%d_%H:%M-%S")

    models_ppo = get_models(args.models)
    states = parse_state_files(Path(args.clipspath).resolve())   

    filters = {
        'sub': args.subjects,
        'ses': args.sessions,
        'level': args.levels,
        'scene': args.scenes
    }

    filtered_states = filter_states(states, filters)
    info_scenes = get_mastersheet(args.mastersheet)

    for i, ppo_row in models_ppo.iterrows():
        # Get all states to process with current model
        states_to_process = [(y, row_state) for y, row_state in sorted(filtered_states.iterrows())]
        
        if args.verbose:
            print(f"Processing {len(states_to_process)} states with model {ppo_row['name_models']}")
        
        # Create a progress bar description
        desc = f"Processing model {ppo_row['name_models']}"
        
        # Use tqdm_joblib to create a progress bar for parallel processing
        with tqdm_joblib(tqdm(total=len(states_to_process), desc=desc, disable=not args.verbose)) as progress_bar:
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(process_single_state)(row_state, ppo_row, info_scenes, args) 
                for _, row_state in states_to_process
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract .state files from Mario dataset and information about the scenes")
    parser.add_argument(
        "-cp",
        "--clipspath",
        default='sourcedata/scenes_clips',
        type=str,
        help="Data path to look for the .state files and .mp4. Should contain replays/ (for .mp4) and scene_clips (for .state)",
    )
    parser.add_argument(
        "-rp",
        "--replayspath",
        default='sourcedata/replays',
        type=str,
        help="Data path to look for the .state files and .mp4. Should contain replays/ (for .mp4) and scene_clips (for .state)",
    )
    parser.add_argument(
        "-md",
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
        help="Path to the mastersheet.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default='outputdata/',
        type=str,
        help="Path to the derivatives folder, where the outputs will be saved.",
    )
    parser.add_argument(
        "-st",
        "--stimuli",
        default=None,
        type=str,
        help="Path to the stimuli folder containing the game ROMs. Defaults to <datapath>/stimuli if not specified.",
    )
    parser.add_argument( 
        '-sub',
        '--subjects',
        nargs='+',
        default=None,
        help='List of subjects to process (e.g., sub-01 sub-02). If not specified, all subjects are processed.'
    )
    parser.add_argument( 
        '-ses',
        '--sessions',
        nargs='+', 
        default=None,
        help='List of sessions to process (e.g., ses-001 ses-002). If not specified, all sessions are processed.'
    )
    parser.add_argument( 
        '-l', 
        '--levels',
        nargs='+', 
        default=None,
        help='List of levels to process (e.g., w1l1 w1l2). If not specified, all sessions are processed.'
    )
    parser.add_argument( 
        '-scn',
        '--scenes',
        nargs='+', 
        default=None,
        help='List of scenes to process (e.g., scene-1 scene-2). If not specified, all sessions are processed.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )
    parser.add_argument(
        '-j',
        '--n_jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs. Default is -1 (use all available cores).'
    )

    args = parser.parse_args()
    main(args)