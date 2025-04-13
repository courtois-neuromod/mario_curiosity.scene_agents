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
import os.path as op
import pickle
from src.ppo.env import preprocess_frames, complex_movement_to_button_presses
from src.ppo.emulation import add_unused_buttons

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
                    'StateFileName': state_path,
                    'Level': state_path.split('/')[-1].split('_')[-3].split('-')[1],
                    'Scene': state_path.split('/')[-1].split('_')[-2].split('-')[1],
                    'StateClipCode': state_path.split('/')[-1].split('_')[-1].split('-')[1],
                    'TotalFrames': n_frames,
                    'Bk2Filepath': bk2_fname,
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

