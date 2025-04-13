import os
import pandas as pd
from pathlib import Path
import json
import numpy as np
import imageio
from scipy.special import softmax
import argparse
import retro
import os.path as op
import pickle


from mario_replays.utils import reformat_info, make_mp4, make_gif, make_webp, create_sidecar_dict

    
def filter_states(states_df, filters):
    """
    Filtre les états en fonction des arguments fournis.
    
    Args:
        states_df (pd.DataFrame): DataFrame contenant les états.
        args (Namespace): Arguments de la ligne de commande.
    
    Returns:
        pd.DataFrame: DataFrame filtré.
    """

    for key, values in filters.items():
        if values is None:
            pass
        else:
            states_df = states_df[states_df[key].isin(values)]

    return states_df


def get_previous_frames (state, args, prev_frames=-16):
    """
    Extrait les frames précédentes d'un état donné.

    Args:
        state (str): Identifiant de l'état du jeu (ex: "sub-01_ses-001_run-01_level-w1l1_scene-0_clip-00101000000122_beh").

    Returns:
        list: Liste des frames précédentes.
    """
    # get the json correspondind to the state
    json_path = state.replace('.state', '.json')
    json_path = json_path.replace('savestates', 'infos')

    # open json file and extract ClipCode and bk2_filepath
    with open(json_path, 'r') as f:
        data = json.load(f)
        clip_code = data['ClipCode']
        bk2_filepath = data['bk2_filepath']
    
    start_frame = int(clip_code[-7:])

    # get the mp4 filepath from bk2 filepath
    mp4_relative_path = bk2_filepath.replace('.bk2', '.mp4')
    mp4_filepath = os.path.join(args.replayspath, mp4_relative_path)

    return mp4_to_list(mp4_filepath, start_frame, prev_frames)

def mp4_to_list(mp4_filepath, start_frame, num_frames):
    """
    Convertit un fichier mp4 en tableau numpy contenant les images.

    Args:
        mp4_filepath (str): Chemin du fichier mp4.
        start_frame (int): Numéro de la frame de départ.
        num_frames (int): Nombre de frames à extraire. Si négatif, ce sont les frames précédant start_frame qui seront extraites.


    Returns:
        np.ndarray: Liste contenant les images extraites sous forme de Numpy array.
    """

    # Lire le fichier mp4 et extraire les frames
    reader = imageio.get_reader(mp4_filepath)
    frames = []

    end_frame = start_frame + num_frames

    id_frames = np.linspace(start_frame, end_frame, abs(num_frames), dtype=int)
    id_frames = id_frames[::-1] if num_frames < 0 else id_frames

    # Extraire les frames
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
    #print(f"frames : {len(frames)}")
    # Convertir les frames en uint8

    return frames


