"""
emulation.py — Shared emulation logic for PPO and imitation pipelines.

Provides reusable functions for:
- Setting up the retro emulator
- Running the main game loop with a model-specific prediction callback
- Building output file paths
- Saving outputs (bk2, json metadata, videos, variables)
"""

import os
import json
import numpy as np
import os.path as op
from pathlib import Path

import stable_retro as retro

from videogames_utils.metadata import create_sidecar_dict
from videogames_utils.replay import reformat_info
from videogames_utils.video import make_mp4, make_gif, make_webp

from src.ppo.emulation import add_unused_buttons


def setup_emulator(state_path, stimuli_path, record_path):
    """Create and initialize a retro emulator loaded with a given state.

    Parameters
    ----------
    state_path : str
        Path to the .state file to load.
    stimuli_path : str
        Path to the stimuli folder containing the game ROMs.
    record_path : str
        Directory where .bk2 replay files will be recorded.

    Returns
    -------
    emul : retro.RetroEnv
        The initialized emulator instance, ready to step.
    """
    resolved_path = Path(stimuli_path).resolve()
    retro.data.Integrations.add_custom_path(resolved_path)

    emul = retro.make(
        game='SuperMarioBros-Nes',
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        record=record_path,
        render_mode=False,
    )
    emul.load_state(state_path)
    emul.reset()
    return emul


def build_output_paths(beh_folder, entities):
    """Build all output file paths for a given state entity.

    Parameters
    ----------
    beh_folder : str
        Base behavior output folder (e.g. outputdata/<model>/<sub>/<ses>/beh).
    entities : str
        Entity identifier extracted from the state filename.

    Returns
    -------
    dict
        Dictionary with keys: gif, mp4, webp, savestate, ramdump, json, variables, bk2.
    """
    return {
        'gif':       op.join(beh_folder, 'videos', f"{entities}.gif"),
        'mp4':       op.join(beh_folder, 'videos', f"{entities}.mp4"),
        'webp':      op.join(beh_folder, 'videos', f"{entities}.webp"),
        'savestate': op.join(beh_folder, 'savestates', f"{entities}.state"),
        'ramdump':   op.join(beh_folder, 'ramdumps', f'{entities}.npz'),
        'json':      op.join(beh_folder, 'infos', f"{entities}.json"),
        'variables': op.join(beh_folder, 'variables', f"{entities}.json"),
        'bk2':       op.join(beh_folder, 'bk2', f"{entities}.bk2"),
    }


def run_emulation_loop(emul, predict_fn, context_frames, x_max,
                       pred_rate=4, verbose=False):
    """Run the emulation loop until a terminal condition is met.

    The loop steps the emulator forward, calling ``predict_fn`` every
    ``pred_rate`` frames to obtain the next action.

    Parameters
    ----------
    emul : retro.RetroEnv
        An initialized emulator (already loaded with a state and reset).
    predict_fn : callable
        A function ``predict_fn(context_frames) -> list[int]`` that takes
        the current context frames and returns a list of button presses
        (before adding unused buttons).
    context_frames : list[np.ndarray]
        Initial context frames (previous observations for the model).
    x_max : int
        Maximum x-position for the current scene. Exceeding this ends the run.
    pred_rate : int, optional
        How often (in frames) to call the prediction function. Default is 4.
    verbose : bool, optional
        Whether to print debug info each prediction step.

    Returns
    -------
    frames_list : list[np.ndarray]
        All observed frames during the run.
    info_list : list[dict]
        Game info dict at each frame.
    keys_list : list[list]
        Button presses at each frame.
    buttons : list[str]
        Button names from the emulator.
    n_frames : int
        Total number of frames stepped.
    """
    n_frames = 0
    done = False
    frames_list = []
    info_list = []
    keys_list = []
    act = None
    lives = None
    level_layout = None

    while not done:
        # Predict new actions every pred_rate frames
        if not n_frames % pred_rate:
            act = predict_fn(context_frames)
            if verbose:
                print(f"[frame {n_frames}] act: {act}")

        keys = add_unused_buttons(act)
        obs, _rew, _term, _trunc, info = emul.step(keys)

        frames_list.append(obs)
        info_list.append(info)
        keys_list.append(keys)
        buttons = emul.buttons

        if n_frames == 0:
            lives = info.get('lives')
            level_layout = info.get('level_layout')

        context_frames.append(obs)
        done = _term or _trunc

        # Compute player x-scroll position
        if "player_x_posHi" in info and "player_x_posLo" in info:
            xscroll = 255 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
        elif "xscrollHi" in info and "xscrollLo" in info:
            xscroll = 255 * int(info["xscrollHi"]) + int(info["xscrollLo"])
        else:
            xscroll = 0

        new_layout = info.get("level_layout")
        new_lives = info.get('lives')

        if (xscroll > x_max
                or (new_lives is not None and new_lives != lives)
                or (new_layout is not None and new_layout != level_layout)
                or _trunc or _term):
            done = True

        n_frames += 1

    return frames_list, info_list, keys_list, buttons, n_frames


def save_metadata(paths, state_path, path_output, emul_gamename, n_frames,
                  info_list, keys_list, buttons):
    """Save JSON metadata and scene sidecar for a completed run.

    Parameters
    ----------
    paths : dict
        Output paths from ``build_output_paths``.
    state_path : str
        Original .state file path.
    path_output : str
        Base output path for the model.
    emul_gamename : str
        Game name from the emulator.
    n_frames : int
        Total frames in the run.
    info_list : list[dict]
        Game info at each frame.
    keys_list : list[list]
        Button presses at each frame.
    buttons : list[str]
        Button names.

    Returns
    -------
    scene_variables : dict
        Reformatted scene variable data.
    """
    scene_variables = reformat_info(info_list, keys_list, paths['bk2'], buttons)

    parts = state_path.split('/')[-1].split('_')
    metadata = {
        'Model': path_output.split('/')[-4] if len(path_output.split('/')) >= 4 else path_output,
        'StateFileName': '/'.join(state_path.split('/')[-6:]),
        'LevelFullName': parts[-3].split('-')[1] if len(parts) >= 3 else '',
        'Scene': parts[-2].split('-')[1] if len(parts) >= 2 else '',
        'StateClipCode': parts[-1].split('-')[1].split('.')[0] if '-' in parts[-1] else parts[-1].split('.')[0],
        'TotalFrames': n_frames,
        'Bk2Filepath': '/'.join(paths['bk2'].split('/')[-6:]),
        'GameName': emul_gamename,
    }

    scene_sidecar = create_sidecar_dict(scene_variables)
    metadata.update(scene_sidecar)

    os.makedirs(os.path.dirname(paths['json']), exist_ok=True)
    with open(paths['json'], 'w') as f:
        json.dump(metadata, f)

    return scene_variables


def save_optional_outputs(paths, args, frames_list, scene_variables):
    """Save optional outputs (videos, ramdumps, variables) based on CLI flags.

    Parameters
    ----------
    paths : dict
        Output paths from ``build_output_paths``.
    args : argparse.Namespace
        Parsed CLI arguments (checked for save_videos, save_ramdumps, save_variables, video_format).
    frames_list : list[np.ndarray]
        Observed frames.
    scene_variables : dict
        Scene variable data to save.
    """
    if args.save_videos:
        os.makedirs(os.path.dirname(paths['mp4']), exist_ok=True)
        format_fn = {'gif': make_gif, 'mp4': make_mp4, 'webp': make_webp}
        save_fn = format_fn.get(args.video_format, make_mp4)
        save_fn(frames_list, paths[args.video_format])

    if args.save_ramdumps:
        os.makedirs(os.path.dirname(paths['ramdump']), exist_ok=True)

    if args.save_variables:
        os.makedirs(os.path.dirname(paths['variables']), exist_ok=True)
        with open(paths['variables'], 'w') as f:
            json.dump(scene_variables, f)
