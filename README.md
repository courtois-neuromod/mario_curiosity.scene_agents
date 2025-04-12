# mario_curiosity.scene_agents
This repository contains scripts that uses artificial agents to play scenes of Super Mario Bros. from human savestates.

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
cd env/lib/python3.10/site-packagesinvoke 
git clone git@github.com:farama-foundation/stable-retro
cd ../../../..
pip install -e  env/lib/python3.10/site-packages/stable-retro/.
pip install -r requirements_beluga.txt
pip install -e .
pip install datalad
invoke get-scenes-data
export AWS_ACCESS_KEY_ID=<s3_access_key>  AWS_SECRET_ACCESS_KEY=<s3_secret_key>
invoke setup-mario-dataset
python src/mario_scenes/create_clips/create_clips.py -d sourcedata/mario --save_state
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








## Installation instructions

First, make sure you have Python 3.10 installed.
```
python --version
```
or 
```
python3 --version
```
If these lines return Python 3.10 we're good to go.

Else, we suggest installing pyenv to manage several Python versions, and select Python 3.10 for the local directory : 
```

```

Create environment :
```
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```



