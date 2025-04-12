import os
import sys
import h5py
import numpy as np
import argparse
from pathlib import Path


sys.path.append(os.path.join(os.getcwd()))



from load_data import get_mastersheet, get_models, parse_state_files, get_scene, get_xpos_max
from utils import process_state, filter_states


def main(args):

    models_ppo = get_models(args.models)
    states = parse_state_files(Path(args.datapath).resolve())   

    filters = {
        'sub': args.subjects,
        'ses': args.sessions,
        'level': args.levels,
        'scene': args.scenes
    }


    filtered_states = filter_states(states, filters)
    info_scenes = get_mastersheet(args.mastersheet)

    h5_path = os.path.join(args.output, "bk2.h5")
    with h5py.File(h5_path, "a") as h5f:

        for i, ppo_row in models_ppo.iterrows():
            session_grp = h5f.create_group(f"model_{ppo_row['name_models']}")
            for y , row_state in filtered_states.iterrows():


                state = row_state['state_path']
                sub = row_state['sub']
                ses = row_state['ses']
                model  = ppo_row['loaded_models']
                scene = get_scene(state)
                x_max = get_xpos_max(info_scenes, scene)
                path_output = os.path.join(args.output, ppo_row['name_models'].split('.')[0], sub, ses, 'beh')

            
                process_state(state, model, x_max, args.stimuli, path_output , verbose=args.verbose)

                bk2_path = os.path.join(args.output, ppo_row["name_models"], row_state["sub"], row_state["ses"], "beh", "bk2", 'SuperMarioBros-Nes-'+row_state["state_path"].split('/')[-1].replace('.state', '-000000.bk2'))
                print(bk2_path)

                with open(bk2_path, "rb") as f:
                    print('f:', f)
                    bk2_data = f.read()
                    print('bk2_data:', bk2_data)

                bk2_bytes = np.frombuffer(bk2_data, dtype=np.uint8)
                session_grp.create_dataset(bk2_path, data=bk2_bytes, compression="gzip")
                session_grp.attrs["model"] = ppo_row["name_models"]
                session_grp.attrs["state"] = row_state["state_path"]

                os.remove(bk2_path)  # Nettoyage



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
