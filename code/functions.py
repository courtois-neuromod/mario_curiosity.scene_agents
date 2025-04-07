import os
import sys
import torch
import pandas as pd
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import re
import cv2

sys.path.append(os.path.join(os.getcwd()))
from src.models import PPO


def get_models(models_path, device='cpu'):
    """
    Charge les modèles PPO depuis un répertoire donné.
    
    Args:
        models_path (str): Chemin vers le dossier contenant les modèles.
        device (str): 'cpu' ou 'cuda' pour exécuter sur GPU si disponible.
    
    Returns:
        list: Une liste des modèles chargés.
    """

    models = []
    models_names = []
    path_models = []
    for model_name in os.listdir(models_path):
        model_file = os.path.join(models_path, model_name)
        if model_file.endswith('.pt'):
            model = PPO(n_in=4, n_actions=12)
            model.load_state_dict(torch.load(model_file, map_location=device))
            torch.set_grad_enabled(False)
            model.to(device)
            model.eval()
            models_names.append(model_name)
            models.append(model)
            path_models.append(model_file)
    
    return pd.DataFrame({'name_models': models_names,
                        'loaded_models': models,
                        'path_models' : path_models
                            })


def parse_state_files(): # a remplacer avec la fonction de Yann
    """Load the states from the derivatives folder with additional info (sub, ses)."""
    base_folder = Path("derivatives/scene_clips")
    print(f"Loading states from {base_folder}")

    if not base_folder.exists():
        print(f"Target folder does not exist: {base_folder}")
        return pd.DataFrame(columns=['state_path', 'sub', 'ses', 'level', 'scene'])

    else :
        state_files = list(base_folder.glob("sub*/ses*/gamelogs/*.states"))

        data = []
        for file in state_files:
            match = re.search(r"(sub-\d+)_+(ses-\d+).*?level-(\w+).*?(scene-\d+)", file)
            sub, ses, level, scene = match.groups()
            data.append({'state_path': str(file), 'sub': sub, 'ses': ses, 'level': level, 'scene': scene})

        return pd.DataFrame(data) # or return la liste state_files
    
def filter_states(states_df, args):
    """
    Filtre les états en fonction des arguments fournis.
    
    Args:
        states_df (pd.DataFrame): DataFrame contenant les états.
        args (Namespace): Arguments de la ligne de commande.
    
    Returns:
        pd.DataFrame: DataFrame filtré.
    """

    filters = {
        'sub': args.sub,
        'ses': args.ses,
        'level': args.level,
        'scene': args.scene
    }

    for key, values in filters.items():
        if values:
            states_df = states_df[states_df[key].isin(values)]

    return states_df

def create_bk2_folders (models, subjects, sessions, base_path):

    """
    Crée les dossiers nécessaires pour stocker les fichiers BK2.
    
    Args:
        base_path (str): Chemin de base où créer les dossiers.
        models (list): Liste des noms de modèles PPO.
        subjects (list): Liste des sujets.
        sessions (list): Liste des sessions.
    """
    for model in models:
        for sub in subjects:
            for ses in sessions:
                path = os.path.join(base_path, 'artificial_agents', model, sub, ses, 'beh', 'bk2')
                os.makedirs(path, exist_ok=True)

def get_mastersheet(filepath):
    """
    Charge et retourne un mastersheet sous forme de DataFrame.
    
    Args:
        filepath (str): Chemin du fichier CSV ou Excel.
    
    Returns:
        pd.DataFrame: Le mastersheet chargé.
    """
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Format de fichier mastershhet non supporté. Utilisez CSV ou Excel.")

def get_scene(state):
    """
    Extrait le monde, le niveau et la scène depuis l'identifiant de l'état.
    
    Args:
        state (str): Identifiant de l'état du jeu (ex: "sub-01_ses-001_run-01_level-w1l1_scene-0_clip-00101000000122_beh").
    
    Returns:
        str: Chaîne formatée (ex: "w1l1s0").
    """
    try:
        level_part = [s for s in state.split('_') if s.startswith('level-')][0]
        scene_part = [s for s in state.split('_') if s.startswith('scene-')][0]
        world = level_part[6]  # 'w1l1' -> '1'
        level = level_part[8]  # 'w1l1' -> '1'
        scene = scene_part.split('-')[1]  # 'scene-0' -> '0'
        return f"w{world}l{level}s{scene}"
    
    except (IndexError, ValueError):
        return "Unknown Scene from the .state file"

def get_xpos_max (mastersheet, scene):
    """
    Extrait la valeur maximale de Xscroll pour une scène donnée depuis le mastersheet.

    Args:
        mastersheet (pd.DataFrame): DataFrame contenant les informations de la scène.
        scene (str): Identifiant de la scène (ex: "w1l1s0").

    Returns:
        int : Valeur maximale de player_x_pos pour la scène donnée.
    """
    
    scene_data = mastersheet[mastersheet['scene'] == scene]
    if scene_data.empty:
        print(f"Scene {scene} not found in the mastersheet.")
        return None
    xpos_max = scene_data['player_x_pos'].max()

    return  int(xpos_max)



def get_previous_frames (state, prev_frames=16):
    """
    Extrait les frames précédentes d'un état donné.

    Args:
        state (str): Identifiant de l'état du jeu (ex: "sub-01_ses-001_run-01_level-w1l1_scene-0_clip-00101000000122_beh").

    Returns:
        list: Liste des frames précédentes.
    """
    # get the json correspondind to the state
    json_path = state.replace('.states', '.json')
    json_path = json_path.replace('savestates', 'clips')

    print(f"json_path : {json_path}")

    # open json file and extract ClipCode and bk2_filepath
    with open(json_path, 'r') as f:
        data = json.load(f)
        clip_code = data['ClipCode']
        bk2_filepath = data['bk2_filepath']
    
    start_frame = clip_code[:-7]

    # get the mp4 filepath from bk2 filepath
    mp4_filepath = bk2_filepath.replace('.bk2', '.mp4')

    pred_rate = 4
    frame_range = np.linspace(start_frame - prev_frames * pred_rate, start_frame, pred_rate, dtype=int)

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(mp4_filepath)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {mp4_filepath}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # S'assurer que les indices de frame sont valides
    frame_range = np.clip(frame_range, 0, total_frames - 1)

    frames = []

    for frame_idx in frame_range:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Frame {frame_idx} non lue.")
            continue
        # Convertir BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    # Convertir en array numpy (shape: [n_frames, height, width, channels])
    return np.array(frames)
    # récuperer les frames grace a un indexge numpy