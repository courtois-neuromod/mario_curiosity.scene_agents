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
```
invoke get-scenes-data
```
## Get ROM folder

```
invoke setup-mario-game
```

## mario.replays

For a quick start on [mario.replays](https://github.com/courtois-neuromod/mario.replays) go check the [README.md](https://github.com/courtois-neuromod/mario.replays/blob/main/README.md). You will only need the videos replay (.mp4) from this repo. Once you are done with the setup, use:
```
invoke create-replays --save-videos --output <path_to_mario_curiosity.scene_agents_sourcedata/>
``` 
## mario.scenes

Form the [mario.scenes](https://github.com/courtois-neuromod/mario.scenes) repository, you will only need saved states (.state) files. Once your setup in done for this repository, see [README.md](https://github.com/courtois-neuromod/mario.scenes/blob/main/README.md), run:
```
invoke create-clips --save-states --output <path_to_mario_curiosity.scene_agents_sourcedata/>
```

## Instructructions to start on compute canada

Go into the scratch
```
cd scratch
```

Import the modules you need
```
module load python/3.10
module load git-annex
```

Git clone the necessary reposittory
```
git clone git@github.com:courtois-neuromod/mario.scenes.git
git clone git@github.com:courtois-neuromod/mario_curiosity.scene_agents.git
```

Import the mastersheet and the .state through marioscenes
```
cd mario.scenes
python -m venv env
source env/bin/activate
pip install invoke
cd env/lib/python3.10/site-packages
git clone git@github.com:farama-foundation/stable-retro
cd ../../../..
pip install -e  env/lib/python3.10/site-packages/stable-retro/.
pip install -r requirements_beluga.txt
pip install -e .
pip install datalad
invoke get-scenes-data
export AWS_ACCESS_KEY_ID=<s3_access_key>  AWS_SECRET_ACCESS_KEY=<s3_secret_key>
invoke setup-mario-dataset
python code/mario_scenes/create_clips/create_clips.py -d sourcedata/mario --save_state
```

Then go back to scene_agents repo
```
cd ..
cd mario_curiosity.scene_agents
```

Creat and activate the venv
```
python -m venv env
source env/bin/activate
cd env/lib/python3.10/site-packages
git clone git@github.com:farama-foundation/stable-retro
git clone git@github.com:courtois-neuromod/ppo_study.git
git clone git@github.com:MaxStrange/retrowrapper.git
cd retrowrapper
nano setup.py
```

Once you open the setup.py of retrowrapper, you can delete the "gym-retro" in the line 11 (install_requires=["gym-retro"]). Or simply delete the line 11 and copy at the same place install_requires=[].

Continue to setup the venv
```
cd ../../../../..
pip install -e env/lib/python3.10/site-packages/stable-retro/.
pip install -e env/lib/python3.10/site-packages/src/.
pip install -e env/lib/python3.10/site-packages/retrowrapper/.
pip install -r requirements_for_beluga.txt
pip install -e .
```

deacti