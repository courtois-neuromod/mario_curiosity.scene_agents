#!/usr/bin/env python3
"""
main.py — Run PPO or imitation models on .state replays.

Unified entry point for both model types. Loads models, replays each scene
state through the emulator using the model's policy, and saves outputs
(bk2, json, optional videos/variables).

Usage
-----
    # PPO models
    python code/main.py -sub sub-06 -ses ses-001 -l w1l1 -scn scene-1 -j 1

    # Imitation models
    python code/main.py --model-type imitation -sub sub-06 -ses ses-001 --device cpu -j 1
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from scipy.special import softmax
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from src.ppo.env import preprocess_frames, complex_movement_to_button_presses
from src.models import ImitationModel

from load_data import (
    get_ppo_models, get_imitation_models,
    parse_state_files, get_mastersheet, get_scene, get_xpos_max,
)
from utils import get_previous_frames, filter_states
from emulation import (
    setup_emulator, build_output_paths, run_emulation_loop,
    save_metadata, save_optional_outputs,
)

sys.path.append(os.path.join(os.getcwd()))

RNG = np.random.default_rng(seed=1)


# ---------------------------------------------------------------------------
# Prediction callbacks
# ---------------------------------------------------------------------------

def make_ppo_predict_fn(model):
    """Create a PPO prediction callback for the emulation loop.

    Returns a function that takes context frames and returns button presses
    via softmax sampling over the model's action logits.

    Parameters
    ----------
    model : PPO
        A loaded PPO model in eval mode.

    Returns
    -------
    callable
        predict_fn(context_frames) -> list[int]
    """
    def predict_fn(context_frames):
        input_frames = np.stack([preprocess_frames(context_frames[-16:], 4, 4)])
        frames_tensor = torch.tensor(input_frames, dtype=torch.float32, device=torch.device('cpu'))
        logits = model(frames_tensor)[0].detach().cpu().numpy()
        probs = softmax(logits, axis=1)
        actions = [RNG.choice(np.arange(12), p=p) for p in probs]
        actions = [complex_movement_to_button_presses(a) for a in actions]
        return actions[0].tolist()
    return predict_fn


def make_imitation_predict_fn(model, best_thres, device):
    """Create an imitation model prediction callback for the emulation loop.

    Returns a function that preprocesses context frames, runs the model,
    and binarizes predictions using per-button thresholds.

    Parameters
    ----------
    model : ImitationModel
        A loaded imitation model in eval mode.
    best_thres : float or np.ndarray
        Per-button threshold(s) for binarizing predictions.
    device : torch.device
        Device the model is on.

    Returns
    -------
    callable
        predict_fn(context_frames) -> list[int]
    """
    def predict_fn(context_frames):
        try:
            input_frames = np.stack([preprocess_frames(context_frames[-16:], 4, 4)])
        except Exception as e:
            print(f"Error preprocessing frames: {e}. Context length: {len(context_frames)}",
                  file=sys.stdout, flush=True)
            return [0] * 6  # safe fallback: no buttons pressed

        frames_tensor = torch.tensor(input_frames, dtype=torch.float32, device=device)
        with torch.no_grad():
            preds = model(frames_tensor).detach().cpu().numpy()

        actions_bool = preds > best_thres
        return actions_bool[0].astype(int).tolist()

    return predict_fn


# ---------------------------------------------------------------------------
# State processing
# ---------------------------------------------------------------------------

def process_single_state(row_state, model_row, info_scenes, args, model,
                         predict_fn_factory):
    """Process a single state file with the given model.

    Parameters
    ----------
    row_state : dict-like
        Row from the states DataFrame (must have state_path, sub, ses).
    model_row : dict-like
        Row from the models DataFrame (must have name_models).
    info_scenes : pd.DataFrame
        Scenes mastersheet.
    args : argparse.Namespace
        CLI arguments.
    model : torch.nn.Module
        The loaded model.
    predict_fn_factory : callable
        A function that returns a predict_fn for the emulation loop.
    """
    state = row_state['state_path']
    sub = row_state['sub']
    ses = row_state['ses']
    scene = get_scene(state)
    x_max = get_xpos_max(info_scenes, scene)
    path_output = os.path.join(
        args.output, model_row['name_models'].split('.')[0], sub, ses, 'beh'
    )

    # For imitation models, only process states matching the model's subject
    if args.model_type == 'imitation' and sub not in model_row['name_models']:
        return None

    return _run_state(state, args, model, predict_fn_factory, x_max, path_output)


def _run_state(state_path, args, model, predict_fn_factory, x_max, path_output):
    """Run a model on a single .state file and save outputs."""
    entities = state_path.split('/')[-1].split('.')[0]
    paths = build_output_paths(path_output, entities)
    os.makedirs(os.path.dirname(paths['bk2']), exist_ok=True)

    emul = setup_emulator(state_path, args.stimuli, os.path.join(path_output, 'bk2'))
    context_frames = get_previous_frames(state_path, args)
    predict_fn = predict_fn_factory(model)

    frames_list, info_list, keys_list, buttons, n_frames = run_emulation_loop(
        emul, predict_fn, context_frames, x_max, verbose=args.verbose
    )

    scene_variables = save_metadata(
        paths, state_path, path_output, emul.gamename, n_frames,
        info_list, keys_list, buttons
    )
    save_optional_outputs(paths, args, frames_list, scene_variables)

    return {"entities": entities, "frames": n_frames, "json": paths['json'], "bk2": paths['bk2']}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    """Load models and states, then process all combinations in parallel."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models based on type
    if args.model_type == 'ppo':
        models_df = get_ppo_models(args.models, device=str(device))
    else:
        models_df = get_imitation_models(args.models, device=str(device))

    states = parse_state_files(Path(args.clipspath).resolve())
    print(f"Found {len(models_df)} models and {len(states)} states.")

    filters = {
        'sub': args.subjects,
        'ses': args.sessions,
        'level': args.levels,
        'scene': args.scenes,
    }
    filtered_states = filter_states(states, filters)
    info_scenes = get_mastersheet(args.mastersheet)

    for _, model_row in tqdm(models_df.iterrows(), desc="models"):
        model = model_row['loaded_models']

        # For imitation, reload checkpoint to the right device
        if args.model_type == 'imitation':
            model_path = model_row['path_models']
            try:
                print(f"Loading model from {model_path}")
                model = ImitationModel.load_from_checkpoint(
                    model_path, map_location=device, weights_only=False
                )
                model.eval().to(device)
                torch.set_grad_enabled(False)
            except Exception as e:
                print(f"Failed to load model {model_path}: {e}")
                continue

        # Build the predict function factory
        if args.model_type == 'ppo':
            predict_fn_factory = make_ppo_predict_fn
        else:
            best_thres = model_row.get('best_thres')
            predict_fn_factory = lambda m: make_imitation_predict_fn(m, best_thres, device)

        states_to_process = [(y, row) for y, row in sorted(filtered_states.iterrows())]
        print(f"Processing {len(states_to_process)} states with model {model_row['name_models']}")

        desc = f"Processing model {model_row['name_models']}"
        with tqdm_joblib(tqdm(total=len(states_to_process), desc=desc, disable=not args.verbose)):
            Parallel(n_jobs=args.n_jobs)(
                delayed(process_single_state)(
                    row_state, model_row, info_scenes, args, model, predict_fn_factory
                )
                for _, row_state in states_to_process
            )

    print("All done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_models_path(model_type):
    """Return the default models path for the given model type."""
    return os.path.join('sourcedata', 'models', model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PPO or imitation models on .state replays and save outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run PPO models on a specific subject/session/level/scene
  python code/main.py -sub sub-06 -ses ses-001 -l w1l1 -scn scene-1 -j 1

  # Run imitation models on CPU
  python code/main.py --model-type imitation -sub sub-06 -ses ses-001 --device cpu -j 1

  # Run with videos saved as webp
  python code/main.py -sub sub-06 --save_videos --video_format webp
        """,
    )

    # Model type
    parser.add_argument("--model-type", "-mt", default="ppo",
                        choices=["ppo", "imitation"],
                        help="Type of model to run (default: ppo).")

    # Paths
    parser.add_argument("-cp", "--clipspath",
                        default=os.path.join('sourcedata', 'scene_clips'),
                        help="Path to scene_clips directory (.state files).")
    parser.add_argument("-rp", "--replayspath",
                        default=os.path.join('sourcedata', 'replays'),
                        help="Path to replays directory (.mp4 files).")
    parser.add_argument("-md", "--models", default=None,
                        help="Path to models directory. Defaults to sourcedata/models/<model-type>/.")
    parser.add_argument("-ms", "--mastersheet",
                        default=os.path.join('sourcedata', 'scenes_info', 'scenes_mastersheet.csv'),
                        help="Path to the scenes mastersheet (CSV or Excel).")
    parser.add_argument("-o", "--output", default="outputdata/",
                        help="Output directory for results.")
    parser.add_argument("-st", "--stimuli",
                        default=os.path.join('sourcedata', 'mario', 'stimuli'),
                        help="Path to the stimuli folder (game ROMs).")

    # Filters
    parser.add_argument('-sub', '--subjects', nargs='+', default=None,
                        help='Subjects to process (e.g., sub-01 sub-02).')
    parser.add_argument('-ses', '--sessions', nargs='+', default=None,
                        help='Sessions to process (e.g., ses-001 ses-002).')
    parser.add_argument('-l', '--levels', nargs='+', default=None,
                        help='Levels to process (e.g., w1l1 w1l2).')
    parser.add_argument('-scn', '--scenes', nargs='+', default=None,
                        help='Scenes to process (e.g., scene-1 scene-2).')

    # Execution
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output.')
    parser.add_argument('-j', '--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all cores).')
    parser.add_argument("--device", default="cpu",
                        help="Device for model inference (cpu or cuda).")

    # Output options
    parser.add_argument("--save_videos", action="store_true",
                        help="Save playback videos.")
    parser.add_argument("--save_variables", action="store_true",
                        help="Save game variables as JSON.")
    parser.add_argument("--save_states", action="store_true",
                        help="Save full RAM states.")
    parser.add_argument("--save_ramdumps", action="store_true",
                        help="Save RAM dumps.")
    parser.add_argument('--video_format', '-vf', default='mp4',
                        choices=['gif', 'mp4', 'webp'],
                        help='Video format (default: mp4).')

    args = parser.parse_args()

    # Set default models path based on model type if not provided
    if args.models is None:
        args.models = _default_models_path(args.model_type)

    main(args)