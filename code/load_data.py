import os
from pathlib import Path
import pandas as pd
import re
import torch

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

def parse_state_files(states_path):

    """Load the states from the derivatives folder with additional info (sub, ses)."""

    if not states_path.exists():
        print(f"Target folder does not exist: {states_path}")
        return pd.DataFrame(columns=['state_path', 'sub', 'ses', 'level', 'scene'])
    
    else :
        print(f"Loading states from {states_path}")
        state_files = list(states_path.glob("sub*/ses*/beh/savestates/*.state"))

        data = []
        for file in state_files:

            match = re.search(r"(sub-\d{2})_(ses-\d{3})_run-\d{2}_level-(\w{4})_(scene-\d{1,2})_clip-(\d+)\.state", str(file))

            if match:
                sub, ses, level, scene, num_clip = match.groups()
            else:
                print(f"[WARNING] Pas de match pour le fichier : {file}")
            
            data.append({'state_path': str(file), 'sub': sub, 'ses': ses, 'level': level, 'scene': scene, 'num_clip': num_clip})

        return pd.DataFrame(data) # or return la liste state_files
    
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
        world = level_part[7]  # 'w1l1' -> '1'
        level = level_part[9]  # 'w1l1' -> '1'
        scene = scene_part.split('-')[1]  # 'scene-0' -> '0'
        return f"w{world}l{level}s{scene}"
    
    except (IndexError, ValueError):
        return "Unknown Scene from the .state file"

def get_xpos_max (ms, scene):
    """
    Extrait la valeur maximale de Xscroll pour une scène donnée depuis le mastersheet.

    Args:
        mastersheet (pd.DataFrame): DataFrame contenant les informations de la scène.
        scene (str): Identifiant de la scène (ex: "w1l1s0").

    Returns:
        int : Valeur maximale de player_x_pos pour la scène donnée.
    """
    ms.dropna(inplace=True)
    ms['LevelFull'] = 'w' + ms['World'].astype(int).astype(str) + 'l' + ms['Level'].astype(int).astype(str) + 's' + ms['Scene'].astype(int).astype(str)
    xpos_max = int(ms[ms['LevelFull'] == scene]['Exit point'].values)

    return  int(xpos_max)