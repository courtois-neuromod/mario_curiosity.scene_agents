import os
import sys
import h5py
import numpy as np
import argparse
from pathlib import Path

from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib  # Import from the official tqdm-joblib package
import torch
from datetime import datetime
import os.path as op
import json
import pickle
from scipy.special import softmax
import retro
import re
from src.ppo.env import preprocess_frames, complex_movement_to_button_presses
from src.ppo.emulation import add_unused_buttons
from mario_replays.utils import reformat_info, make_mp4, make_gif, make_webp, create_sidecar_dict
from load_data import get_mastersheet, get_models, parse_state_files, get_scene, get_xpos_max
from utils import get_previous_frames, filter_states

sys.path.append(os.path.join(os.getcwd()))

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


def process_state(state_path, args, ppo, x_max, path_output, stimuli, verbose=False):

    entities = state_path.split('/')[-1].split('.')[0]
    beh_folder = path_output
    # Prepare output filenames
    gif_fname       = op.join(beh_folder, 'videos', f"{entities}.gif")
    mp4_fname       = op.join(beh_folder, 'videos', f"{entities}.mp4")
    webp_fname      = op.join(beh_folder, 'videos', f"{entities}.webp")
    savestate_fname = op.join(beh_folder, 'savestates', f"{entities}.state")
    ramdump_fname   = op.join(beh_folder, 'ramdumps', f'{entities}.npz')
    json_fname      = op.join(beh_folder, 'infos', f"{entities}.json")
    variables_fname = op.join(beh_folder, 'variables', f"{entities}.pkl")
    bk2_fname       = op.join(beh_folder, 'bk2', f"{entities}.bk2")
    os.makedirs(os.path.dirname(bk2_fname), exist_ok=True)

    # Setup the environments with Mario
    resolved_path = Path(stimuli).resolve()
    retro.data.Integrations.add_custom_path(resolved_path)
    
    emul = retro.make(game='SuperMarioBros-Nes', 
                      inttype=retro.data.Integrations.CUSTOM_ONLY, 
                      record=op.join(beh_folder, 'bk2'),
                      render_mode=False)
    emul.load_state(state_path)
    emul.reset()
    context_frames = get_previous_frames(state_path, args)

    rng = np.random.default_rng(seed=1)
    pred_rate = 4
    n_frames = 0
    done = False
    frames_list = []
    info_list = []
    keys_list = []

    while not done:
        # Predict new actions
        if not n_frames % pred_rate:
            contexts_frames = [
                preprocess_frames(context_frames[-16:], 4, 4)
                    ]

            input_frames = np.stack(contexts_frames)
            frames_input = torch.tensor(
                    input_frames, dtype=torch.float32, device=torch.device('cpu')
               )
            logits = ppo(frames_input)[0].detach().cpu().numpy()
            probs = softmax(logits, axis=1)
            actions = [rng.choice(np.arange(12), p=p) for p in probs]
            actions = [complex_movement_to_button_presses(a) for a in actions]

            if verbose:
                print("Actions : ", actions)
            act = actions[0].tolist()

        keys = add_unused_buttons(act)
        obs, _rew, _term, _trunc, info = emul.step(keys)
        frames_list.append(obs)
        info_list.append(info)
        keys_list.append(keys)
        buttons = emul.buttons


        if n_frames == 0:
            lives = info['lives']
            level_layout = info['level_layout']

        context_frames.append(obs)
        done = _term
        xscroll = 255 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
        new_layout = info["level_layout"]
        new_lives = info['lives']

        if ( xscroll > x_max
            or new_lives != lives
            or new_layout != level_layout
            or _trunc or _term
            ):
                done = True
                
        n_frames += 1
    
    scene_variables = reformat_info(info_list, keys_list, bk2_fname, buttons)

    # Generate json
    metadata = {
                    'Model': path_output.split('/')[-4],
                    'StateFileName': '/'.join(state_path.split('/')[-6:]),
                    'LevelFullName': state_path.split('/')[-1].split('_')[-3].split('-')[1],
                    'Scene': state_path.split('/')[-1].split('_')[-2].split('-')[1],
                    'StateClipCode': state_path.split('/')[-1].split('_')[-1].split('-')[1].split('.')[0],
                    'TotalFrames': n_frames,
                    'Bk2Filepath': '/'.join(bk2_fname.split('/')[-6:]),
                    'GameName': emul.gamename,
                }
    scene_sidecar = create_sidecar_dict(scene_variables)
    enriched_metadata = metadata.copy()
    enriched_metadata.update(scene_sidecar)
    os.makedirs(os.path.dirname(json_fname), exist_ok=True)

    with open(json_fname, 'w') as json_file:    
        json.dump(enriched_metadata, json_file)

    # Generate files
    if args.save_videos:
        os.makedirs(os.path.dirname(gif_fname), exist_ok=True)
        if args.video_format == 'gif':
            make_gif(frames_list, gif_fname)
        elif args.video_format == 'mp4':
            make_mp4(frames_list, mp4_fname)
        elif args.video_format == 'webp':
            make_webp(frames_list, webp_fname)
    if args.save_ramdumps:
        os.makedirs(os.path.dirname(ramdump_fname), exist_ok=True)
        np.savez_compressed(ramdump_fname, replay_states[start_idx:end_idx])
    if args.save_variables:
        os.makedirs(os.path.dirname(variables_fname), exist_ok=True)
        with open(variables_fname, 'wb') as f:
            pickle.dump(scene_variables, f)


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

    for i, ppo_row in tqdm(models_ppo.iterrows()):
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
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save the playback video file (.mp4).",
    )
    parser.add_argument(
        "--save_variables",
        action="store_true",
        help="Save the variables file (.npz) that contains game variables.",
    )
    parser.add_argument(
        "--save_states",
        action="store_true",
        help="Save full RAM state at each frame into a *_states.npy file.",
    )
    parser.add_argument(
        "--save_ramdumps",
        action="store_true",
        help="Save RAM dumps at each frame into a *_ramdumps.npy file.",
    )
    parser.add_argument(
        '--video_format', '-vf', default='mp4',
        choices=['gif', 'mp4', 'webp'],
        help='Video format to save (default: mp4).'
    )
    args = parser.parse_args()
    main(args)