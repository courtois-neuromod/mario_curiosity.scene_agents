import os
import torch
import pandas as pd
from pathlib import Path
import json
import numpy as np
import imageio
from scipy.special import softmax
import argparse
import retro

from src.ppo.env import preprocess_frames, complex_movement_to_button_presses
from src.ppo.emulation import add_unused_buttons
from src.models import PPO

from load_data import get_mastersheet, get_models, parse_state_files, get_scene, get_xpos_max



    
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


def get_previous_frames (state, prev_frames=-16):
    """
    Extrait les frames précédentes d'un état donné.

    Args:
        state (str): Identifiant de l'état du jeu (ex: "sub-01_ses-001_run-01_level-w1l1_scene-0_clip-00101000000122_beh").

    Returns:
        list: Liste des frames précédentes.
    """
    # get the json correspondind to the state
    json_path = state.replace('.state', '.json')
    json_path = json_path.replace('savestates', 'clips')

    # open json file and extract ClipCode and bk2_filepath
    with open(json_path, 'r') as f:
        data = json.load(f)
        clip_code = data['ClipCode']
        bk2_filepath = data['bk2_filepath']
    
    start_frame = int(clip_code[-7:])

    # get the mp4 filepath from bk2 filepath
    mp4_relative_path = bk2_filepath.replace('.bk2', '.mp4')
    mp4_filepath = os.path.join('/home', 'hugo', 'github', 'mario.scenes', 'data', 'mario', 'derivatives', 'replays', mp4_relative_path)

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
    print(f"frames : {len(frames)}")
    # Convertir les frames en uint8

    return frames

def process_state(row_state, ppo_row, info_scene, stimuli, verbose=False):

    state = row_state['state_path']

    sub = row_state['sub']
    ses = row_state['ses']
    model  = ppo_row['loaded_models']

    path_output = os.path.join(os.getcwd(), 'outputdata', ppo_row['name_models'].split('.')[0], sub, ses, 'beh')


    os.makedirs(path_output, exist_ok=True)

    scene = get_scene(state)
    max_xscroll = get_xpos_max(info_scene, scene)

    # Setup the environments with Mario
    resolved_path = Path(stimuli).resolve()
    retro.data.Integrations.add_custom_path(resolved_path)
    
    emul = retro.make(game='SuperMarioBros-Nes', 
                      inttype=retro.data.Integrations.CUSTOM_ONLY, 
                      record=path_output)
    emul.load_state(state)
    emul.reset()

    context_frames = get_previous_frames(state)

    rng = np.random.default_rng(seed=1)
    pred_rate = 4
    n_frames = 0
    done = False

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
            logits = model(frames_input)[0].detach().cpu().numpy()
            probs = softmax(logits, axis=1)
            actions = [rng.choice(np.arange(12), p=p) for p in probs]
            actions = [complex_movement_to_button_presses(a) for a in actions]
            if verbose:
                print("Actions : ", actions)
            act = actions[0].tolist()
            a = add_unused_buttons(act)
            print(a)

            obs, _rew, _term, _trunc, info = emul.step(add_unused_buttons(act))

            if n_frames == 0:
                lives = info['lives']
                level_layout = info['level_layout']

            context_frames.append(obs)
            done = _term
            xscroll = 255 * int(info["player_x_posHi"]) + int(info["player_x_posLo"])
            new_layout = info["level_layout"]
            new_lives = info['lives']

            if ( xscroll > max_xscroll 
                or new_lives != lives
                or new_layout != level_layout
                or _trunc or _term
                ):
                    done = True
                
        n_frames += 1

    #bk2_fname = f'{state.split("/")[-1].replace(".state", "-000000.bk2")}'
    #new_bk2_path = os.path.join(path_output, bk2_fname)
    #os.rename(emul.get_bk2_path(), new_bk2_path)
