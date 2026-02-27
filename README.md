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

# Run agents

Use the `run-agents` invoke task to replay scene savestates with artificial agents (PPO or imitation models).

## Basic usage
```bash
# Run PPO agents on a specific subject and session
invoke run-agents --subjects "sub-06" --sessions "ses-001"

# Run imitation agents
invoke run-agents --model-type imitation --subjects "sub-06" --sessions "ses-001"

# Filter by level and scene
invoke run-agents --subjects "sub-06" --sessions "ses-001" --levels "w1l1" --scenes "scene-1"

# Save playback videos as webp
invoke run-agents --subjects "sub-06" --save-videos --video-format webp
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | `ppo` | Model type: `ppo` or `imitation` |
| `--subjects` | all | Space-separated subjects (e.g., `"sub-01 sub-06"`) |
| `--sessions` | all | Space-separated sessions (e.g., `"ses-001 ses-002"`) |
| `--levels` | all | Space-separated levels (e.g., `"w1l1 w1l2"`) |
| `--scenes` | all | Space-separated scenes (e.g., `"scene-1 scene-2"`) |
| `--n-jobs` | `-1` | Number of parallel jobs (`-1` = all cores) |
| `--device` | `cpu` | Device for inference (`cpu` or `cuda`) |
| `--verbose` | off | Enable verbose output |
| `--save-videos` | off | Save playback videos |
| `--save-variables` | off | Save game variables as JSON |
| `--video-format` | `mp4` | Video format: `gif`, `mp4`, or `webp` |

Outputs are saved to `outputdata/<model_name>/<subject>/<session>/beh/` with subdirectories for `bk2/` (replay files), `infos/` (JSON metadata), and optionally `videos/` and `variables/`.
