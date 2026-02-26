# mario_curiosity.scene_agents
This repository contains scripts that uses artificial agents to play scenes of Super Mario Bros. from human savestates.

# Installation instructions

## Clone of the git repo
```
git clone git@github.com:courtois-neuromod/mario_curiosity.scene_agents.git
cd mario_curiosity.scene_agents
```

## Check Python version
Use `python3 --version` to check the current python version you are using. If it is not 3.10, you need to install it and set it as local version with:
```
pyenv install 3.10.14
pyenv local 3.10.14
```
## Setup venv
```
pip install invoke
invoke setup-env
```
invoke setup-env will:

- Create a virtual environment in ./env/
- Install the mario_curiosity.scene_agents package in editable mode
- Install retrowrapper with few patchs

# Get sourcedata

To play with this repo you need the sourcedata. You can get them with the [mario.scenes](https://github.com/courtois-neuromod/mario.scenes) and [mario.replays](https://github.com/courtois-neuromod/mario.replays) repos, as well with the invoke functions. For the two repo, we recommende you to clone them outside of mario_curiosity.scene_agents

## Scenes data 
Download scene metadata from Zenodo. All files are saved to sourcedata/scenes_info/. 
```
invoke get-scenes-data
```
## Get ROM folder
Download the Mario game using datalad. The Game ROM is installed into sourcedata/mario.stimuli/.
```
invoke setup-mario-game
```

## mario.replays

For a quick start on [mario.replays](https://github.com/courtois-neuromod/mario.replays) go check the [README.md](https://github.com/courtois-neuromod/mario.replays/blob/main/README.md). You will only need the videos replay (.mp4) from this repo. Once you are done with the setup, use:
```
invoke create-replays --save-videos --output <path_to_mario_curiosity.scene_agents_sourcedata/>
``` 
The `--ouput` path should point to sourcedata/replays/.
## mario.scenes

Form the [mario.scenes](https://github.com/courtois-neuromod/mario.scenes) repository, you will only need saved states (.state) files  and infos/. Once your setup in done for this repository, see [README.md](https://github.com/courtois-neuromod/mario.scenes/blob/main/README.md), run:
```
invoke create-clips --save-states --output <path_to_mario_curiosity.scene_agents_sourcedata/>
```
The `--ouput` path should point to sourcedata/scene_clips/.
