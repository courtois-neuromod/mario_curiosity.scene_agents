"""
load_data.py — Data loading utilities for models, states, and scene metadata.

Provides functions to:
- Load PPO and imitation models from disk
- Parse .state files into a structured DataFrame
- Load the scenes mastersheet
- Extract scene identifiers and x-position limits
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models import PPO, ImitationModel


def get_ppo_models(models_path, device='cpu'):
    """Load PPO models (.pt files) from a directory.

    Parameters
    ----------
    models_path : str
        Path to the folder containing .pt model files.
    device : str, optional
        Device to load models onto ('cpu' or 'cuda'). Default 'cpu'.

    Returns
    -------
    pd.DataFrame
        Columns: name_models, loaded_models, path_models, best_thres (always None for PPO).
    """
    rows = []
    for root, _, files in os.walk(models_path):
        for file in files:
            if file.endswith('.pt'):
                model_path = os.path.join(root, file)
                model = PPO(n_in=4, n_actions=12)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device).eval()
                rows.append({
                    'name_models': file,
                    'loaded_models': model,
                    'path_models': model_path,
                    'best_thres': None,
                })

    return pd.DataFrame(rows, columns=['name_models', 'loaded_models', 'path_models', 'best_thres'])


def get_imitation_models(models_path, device='cpu'):
    """Load imitation models (.ckpt files) from a directory.

    Each subdirectory is expected to contain one or more .ckpt checkpoints
    and a ``best_thresholds_f1_score.npy`` file with per-button thresholds.

    Parameters
    ----------
    models_path : str
        Path to the folder containing imitation model subdirectories.
    device : str, optional
        Device to load models onto ('cpu' or 'cuda'). Default 'cpu'.

    Returns
    -------
    pd.DataFrame
        Columns: name_models, loaded_models, path_models, best_thres.
    """
    rows = []
    for root, _, files in os.walk(models_path):
        ckpt_files = [f for f in files if f.endswith('.ckpt')]
        if not ckpt_files:
            continue

        # Sort checkpoints by (epoch, step)
        pairs = []
        for f in ckpt_files:
            epoch = int(re.search(r"epoch=(\d+)", f).group(1))
            step = int(re.search(r"step=(\d+)", f).group(1))
            pairs.append((epoch, step))
        pairs.sort()

        # Load per-button thresholds
        thresholds = np.load(os.path.join(root, 'best_thresholds_f1_score.npy'))
        print(f"Found thresholds {thresholds} in {root}")

        # Extract subject from directory name
        sub_match = re.search(r"sub-(\d{2})_", root)
        sub_prefix = sub_match.group(0) if sub_match else ''

        for i, (epoch, step) in enumerate(pairs):
            ckpt_name = f'epoch={epoch}-step={step}.ckpt'
            ckpt_path = os.path.join(root, ckpt_name)
            model = ImitationModel.load_from_checkpoint(ckpt_path, weights_only=False)
            torch.set_grad_enabled(False)
            model.to(device).eval()

            print(f"Loaded model {ckpt_name} with threshold {thresholds[i]}")
            rows.append({
                'name_models': sub_prefix + ckpt_name,
                'loaded_models': model,
                'path_models': ckpt_path,
                'best_thres': thresholds[i],
            })

    return pd.DataFrame(rows, columns=['name_models', 'loaded_models', 'path_models', 'best_thres'])


def get_models(models_path, device='cpu'):
    """Load all models (PPO and imitation) from a directory tree.

    This is a backward-compatible wrapper that calls both
    ``get_ppo_models`` and ``get_imitation_models``.

    Parameters
    ----------
    models_path : str
        Root model directory to walk.
    device : str, optional
        Device to load models onto. Default 'cpu'.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all loaded models.
    """
    ppo_df = get_ppo_models(models_path, device)
    imit_df = get_imitation_models(models_path, device)
    return pd.concat([ppo_df, imit_df], ignore_index=True)


def parse_state_files(states_path):
    """Parse .state files from a directory into a structured DataFrame.

    Expects the directory structure:
    ``<states_path>/sub-XX/ses-YYY/beh/savestates/*.state``

    Parameters
    ----------
    states_path : Path or str
        Root directory containing the state files.

    Returns
    -------
    pd.DataFrame
        Columns: state_path, sub, ses, level, scene, num_clip.
    """
    states_path = Path(states_path)
    if not states_path.exists():
        print(f"Target folder does not exist: {states_path}")
        return pd.DataFrame(columns=['state_path', 'sub', 'ses', 'level', 'scene'])

    print(f"Loading states from {states_path}")
    state_files = list(states_path.glob("sub*/ses*/beh/savestates/*.state"))

    data = []
    pattern = re.compile(
        r"(sub-\d{2})_(ses-\d{3})_run-\d{2}_level-(\w{4})_(scene-\d{1,2})_clip-(\d+)\.state"
    )
    for file in state_files:
        match = pattern.search(str(file))
        if match:
            sub, ses, level, scene, num_clip = match.groups()
            data.append({
                'state_path': str(file),
                'sub': sub, 'ses': ses,
                'level': level, 'scene': scene,
                'num_clip': num_clip,
            })
        else:
            print(f"[WARNING] No match for file: {file}")

    return pd.DataFrame(data)


def get_mastersheet(filepath):
    """Load the scenes mastersheet (CSV or Excel).

    Parameters
    ----------
    filepath : str
        Path to the mastersheet file.

    Returns
    -------
    pd.DataFrame
        The loaded mastersheet.
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported mastersheet format: {filepath}. Use CSV or Excel.")


def get_scene(state_path):
    """Extract a scene identifier (e.g. 'w1l1s0') from a state file path.

    Parameters
    ----------
    state_path : str
        Path or filename of a .state file.

    Returns
    -------
    str
        Scene identifier like 'w1l1s0', or an error message if parsing fails.
    """
    try:
        level_part = [s for s in state_path.split('_') if s.startswith('level-')][0]
        scene_part = [s for s in state_path.split('_') if s.startswith('scene-')][0]
        world = level_part[7]   # 'level-w1l1' -> '1'
        level = level_part[9]   # 'level-w1l1' -> '1'
        scene = scene_part.split('-')[1]
        return f"w{world}l{level}s{scene}"
    except (IndexError, ValueError):
        return "Unknown scene from .state file"


def get_xpos_max(mastersheet, scene):
    """Get the maximum x-position (exit point) for a scene.

    Parameters
    ----------
    mastersheet : pd.DataFrame
        The scenes mastersheet with World, Level, Scene, and 'Exit point' columns.
    scene : str
        Scene identifier like 'w1l1s0'.

    Returns
    -------
    int
        Maximum x-position for the scene.
    """
    ms = mastersheet.dropna().copy()
    ms['LevelFull'] = (
        'w' + ms['World'].astype(int).astype(str)
        + 'l' + ms['Level'].astype(int).astype(str)
        + 's' + ms['Scene'].astype(int).astype(str)
    )
    return int(ms[ms['LevelFull'] == scene]['Exit point'].values[0])