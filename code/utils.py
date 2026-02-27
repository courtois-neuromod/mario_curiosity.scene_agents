"""
utils.py — Utility functions for state filtering and frame extraction.

Provides helpers for:
- Filtering state DataFrames by subject, session, level, scene
- Extracting previous context frames from replay videos
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import imageio


def filter_states(states_df, filters):
    """Filter a states DataFrame by subject, session, level, and/or scene.

    Parameters
    ----------
    states_df : pd.DataFrame
        DataFrame with columns like 'sub', 'ses', 'level', 'scene'.
    filters : dict
        Keys are column names, values are lists of allowed values (or None to skip).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    for key, values in filters.items():
        if values is not None:
            states_df = states_df[states_df[key].isin(values)]
    return states_df


def get_previous_frames(state, args, prev_frames=-16):
    """Extract context frames preceding a given state from its replay video.

    Reads the JSON sidecar of the state to find the corresponding .mp4 replay,
    then extracts ``abs(prev_frames)`` frames ending at the state's start frame.

    Parameters
    ----------
    state : str
        Path to the .state file.
    args : argparse.Namespace
        Must contain ``replayspath`` attribute.
    prev_frames : int, optional
        Number of previous frames to extract (negative = before start). Default -16.

    Returns
    -------
    list[np.ndarray]
        List of extracted frames as numpy arrays.
    """
    # Get the json corresponding to the state
    json_path = state.replace('.state', '.json')
    json_path = json_path.replace('savestates', 'infos')

    with open(json_path, 'r') as f:
        data = json.load(f)
        clip_code = data['ClipCode']
        run = data['Run']
        sub = data['Subject']
        ses = data['Session']

    start_frame = int(clip_code[-7:])
    bk2_id = clip_code[-9:-7]

    events_path = json_path.replace('scene_clips', 'mario').replace('beh/infos', 'func')
    events_name = f'sub-{sub}_ses-{ses}_task-mario_run-{run}_events.tsv'

    try:
        events_file = pd.read_csv(
            events_path.replace(json_path.split('/')[-1], events_name), sep="\t"
        )
    except FileNotFoundError:
        # Handle known naming exceptions for specific subjects/sessions
        if sub == '06':
            if ses == '003' and run == '05':
                events_name = f'sub-{sub}_ses-{ses}_20210709-154422_task-mario_run-{run}_events.tsv'
            elif ses == '004' and run == '06':
                events_name = f'sub-{sub}_ses-{ses}_20210716-151420_task-mario_run-{run}_events.tsv'
        elif sub == '03' and ses == '003' and run == '06':
            events_name = f'sub-{sub}_ses-{ses}_20211104-184232_task-mario_run-{run}_events.tsv'
        elif sub == '02' and ses == '002' and (run == '06' or run == '07'):
            events_name = f'sub-{sub}_ses-{ses}_20210708-133807_task-mario_run-{run}_events.tsv'

        events_file = pd.read_csv(
            events_path.replace(json_path.split('/')[-1], events_name), sep="\t"
        )

    try:
        events_file_clean = events_file['stim_file'].dropna()
        bk2_path = events_file_clean.iloc[int(bk2_id)].split('/')[-1]
    except Exception as e:
        print("############################################################", file=sys.stderr, flush=True)
        print(f"Error reading BK2 path for {events_path} at bk2_id {bk2_id}", file=sys.stderr, flush=True)
        print(f"Error type: {type(e).__name__}, message: {e}", file=sys.stderr, flush=True)
        print("#############################################################", file=sys.stderr, flush=True)
        raise

    mp4_file = bk2_path.replace('.bk2', '.mp4')
    mp4_path = json_path.replace('scene_clips', 'replays').replace('infos', 'videos')
    mp4_relative_path = mp4_path.replace(json_path.split('/')[-1], mp4_file)
    mp4_filepath = os.path.join(args.replayspath, mp4_relative_path)

    return _mp4_to_frames(mp4_filepath, start_frame, prev_frames)


def _mp4_to_frames(mp4_filepath, start_frame, num_frames):
    """Extract a range of frames from an mp4 video file.

    Parameters
    ----------
    mp4_filepath : str
        Path to the mp4 file.
    start_frame : int
        Starting frame index.
    num_frames : int
        Number of frames to extract. If negative, extracts frames
        *before* start_frame.

    Returns
    -------
    list[np.ndarray]
        Extracted frames as numpy arrays.
    """
    try:
        reader = imageio.get_reader(mp4_filepath)
    except Exception as e:
        print("############################################################", file=sys.stderr, flush=True)
        print(f"Error reading mp4 file: {mp4_filepath}", file=sys.stderr, flush=True)
        print(f"Error type: {type(e).__name__}, message: {e}", file=sys.stderr, flush=True)
        print("#############################################################", file=sys.stderr, flush=True)
        raise

    end_frame = start_frame + num_frames
    id_frames = np.linspace(start_frame, end_frame, abs(num_frames), dtype=int)
    if num_frames < 0:
        id_frames = id_frames[::-1]

    frames = []
    for i in id_frames:
        try:
            frame = reader.get_data(i)
            frames.append(np.array(frame))
        except IndexError:
            print(f"Frame {i} not found in the video.")
            break
        except Exception as e:
            print(f"Error reading frame {i}: {e}")
            break

    reader.close()
    return frames
