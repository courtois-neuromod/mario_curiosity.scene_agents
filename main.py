import os
import sys
import retro
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from scipy.special import softmax
import pandas as pd


sys.path.append(os.path.join(os.getcwd()))

from scripts.functions import get_models, parse_state_files, filter_states, get_mastersheet, get_scene, get_xpos_max, get_previous_frames
from src.ppo.env import preprocess_frames, complex_movement_to_button_presses
from src.ppo.emulation import add_unused_buttons

<<<<<<< HEAD
def process_state(row_state, model, info_scene):

    state = row_state[0]
    sub = row_state[1]
    ses = row_state[2]

    path = os.path.join(os.getcwd, 'datasource', 'artificial_agents', model, sub, ses, 'beh', 'bk2')

    if not os.path.exists(path):
        # Create the directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
=======
def process_state(row_state, ppo_row, info_scene):

    state = row_state['state_path']
    sub = row_state['sub']
    ses = row_state['ses']
    model  = ppo_row['loaded_models']
    print('state', state)

    path_output = os.path.join(os.getcwd(), 'outputdata', ppo_row['name_models'], sub, ses, 'beh', 'bk2')

    if not os.path.exists(path_output):
        # Create the directory if it doesn't exist
            os.makedirs(path_output, exist_ok=True)
>>>>>>> 06fdbf0accfead723acdada50f01dafbe4085dc8

    scene = get_scene(state)
    max_xscroll = get_xpos_max(info_scene, scene)

    # Setup the environments with Mario
<<<<<<< HEAD
    integration_path = 'custom_integrations'
    resolved_path = Path(integration_path).resolve()
    retro.data.Integrations.add_custom_path(resolved_path)


    
    path_bk2 = os.path.join(os.getcwd(),'artificial_agents', model, sub, ses, 'beh', 'bk2')
    emul = retro.make(game='SuperMarioBros-Nes', 
                      state=state, 
                      inttype=retro.data.Integrations.CUSTOM_ONLY, 
                      record=path_bk2)
    emul.reset()

    context_frames = get_previous_frames(state)
=======
    integration_path = '.'
    resolved_path = Path(integration_path).resolve()
    retro.data.Integrations.add_custom_path(resolved_path)
    print(resolved_path)
    
    emul = retro.make(game='SuperMarioBros-Nes', 
                      state=state, 
                      inttype=retro.data.Integrations.CUSTOM_ONLY, 
                      record=path_output)
    emul.reset()

    context_frames = get_previous_frames(state)


    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        
    ''' # Remplir la grille
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            print(idx)
            ax = axes[i, j]
            ax.imshow(context_frames[idx, :, : ,:])
            ax.axis("off")  # Toujours cacher les axes, même s’il n’y a pas d’image

    plt.tight_layout()
    plt.show()
    plt.close()
    '''

>>>>>>> 06fdbf0accfead723acdada50f01dafbe4085dc8
    rng = np.random.default_rng(seed=1)
    pred_rate = 4
    n_frames = 0
    done = False

    while not done:

        # Predict new actions
        if not n_frames % pred_rate:
            contexts_frames = [
<<<<<<< HEAD
                preprocess_frames(cf[-16:], 4, 4) for cf in context_frames
=======
                preprocess_frames(context_frames[-16:], 4, 4)
>>>>>>> 06fdbf0accfead723acdada50f01dafbe4085dc8
                    ]

            input_frames = np.stack(contexts_frames)
            frames_input = torch.tensor(
                    input_frames, dtype=torch.float32, device=torch.device('cpu')
               )
            logits = model(frames_input)[0].detach().cpu().numpy()
            probs = softmax(logits, axis=1)
            actions = [rng.choice(np.arange(12), p=p) for p in probs]
            actions = [complex_movement_to_button_presses(a) for a in actions]
<<<<<<< HEAD

            act = actions.tolist()
            obs, _rew, _term, info = emul.step(add_unused_buttons(act))
=======
            print(actions)
            act = actions[0].tolist()
            a = add_unused_buttons(act)
            print(a)

            obs, _rew, _term, _trunc, info = emul.step(add_unused_buttons(act))
>>>>>>> 06fdbf0accfead723acdada50f01dafbe4085dc8

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
<<<<<<< HEAD
=======
                or _trunc or _term
>>>>>>> 06fdbf0accfead723acdada50f01dafbe4085dc8
                ):
                    done = True
                
        n_frames += 1

<<<<<<< HEAD
def main(models_path, data_source_path):
=======
def main(models_path, data_source_path, args):
>>>>>>> 06fdbf0accfead723acdada50f01dafbe4085dc8

    models_ppo = get_models(models_path)

    states = parse_state_files()
<<<<<<< HEAD
    print(states.head())
    filtered_states = filter_states(states)

    #create_bk2_folders(models_path, states_path) # a intégrer dans process_state
    info_scene = get_mastersheet(path='/home/hugo/simexp/mario.scenes/resources/scenes_mastersheet.csv')


    for ppo in models_ppo:
        for index, state_row in states.iterrows():
            process_state(ppo, state_row, info_scene)
    


models_path = os.path.join(os.getcwd(), 'PPO_checkpoints_for_hugo')
data_source_path = os.path('home', 'hugo', 'github', 
                           'mario.scenes', 'data', 'mario',
                             'deravatives',  'scenes_clips')
main(models_path)
=======
    filtered_states = filter_states(states, args)

    #create_bk2_folders(models_path, states_path) # a intégrer dans process_state
    info_scene = get_mastersheet(filepath='/home/hugo/github/mario.scenes/resources/scenes_mastersheet.csv')

    for i, ppo_row in models_ppo.iterrows():
        for y , state_row in filtered_states.iterrows():
            process_state(state_row, ppo_row, info_scene)
            break
        break

    

models_path = os.path.join(os.getcwd(), 'PPO_checkpoints_for_hugo')
data_source_path = ""
agrs = 'sub-05'
main(models_path, data_source_path, agrs)
>>>>>>> 06fdbf0accfead723acdada50f01dafbe4085dc8





""""
def process_state(model, state, INFO_SCENES):
    blabla process that stuff

def main(args):
    subjects = args.subjects
    models = args.models
    levels = args.levels
    sourcedata_path = args.sourcedata_path
    outputdata_path = args.outputdata_path

    INFO_SCENES = get_mastersheet(path='/home/hugo/simexp/mario.scenes/resources/scenes_mastersheet.csv')

    bk2_files = list(root_folder.rglob("*.bk2"))
    
    for subj in subjects:
        for level in levels:
            for model in models:
                # Filter the list of bk2
                bk2_files_filtered = [
                    bk2_file for bk2_file in bk2_files
                    if f"{subj}" in str(bk2_file) and f"{level}" in str(bk2_file)
                ]
                for bk2_file in bk2_files_filtered:
                    # Process each file in the selected bk2s
                    process_state(model, state, INFO_SCENES)


"""
